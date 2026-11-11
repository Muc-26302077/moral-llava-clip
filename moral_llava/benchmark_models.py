"""Final VLGuard conversion and native image/prompt preprocessing."""
from pathlib import Path
from .common import read, sha

def dump(path,value):
    import json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")

def load_llava(base_model):
    BASE=base_model
    adapter=None
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from moral_llava.train import configure_processor
    model=LlavaForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa',low_cpu_mem_usage=True,local_files_only=True)
    source=adapter if adapter and (Path(adapter)/'processor_config.json').exists() else BASE
    processor=AutoProcessor.from_pretrained(source,local_files_only=True)
    metadata={}
    configure_processor(processor,model.config,padding_side='left')
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    return model,processor,metadata

def load_vlguard(name, base_model, source, audit_path):
    BASE=base_model
    """Map all original LLaVA weights explicitly onto HF; copy frozen CLIP only."""
    import torch
    from transformers import AutoTokenizer
    model,processor,_=load_llava(base_model)
    source=Path(source)
    original_config=read(source/'config.json')
    expected_config={'image_aspect_ratio':'pad','mm_vision_tower':'openai/clip-vit-large-patch14-336',
                     'mm_vision_select_layer':-2,'mm_vision_select_feature':'patch',
                     'mm_use_im_start_end':False,'rope_scaling':None,'vocab_size':32000}
    if any(original_config.get(k)!=v for k,v in expected_config.items()):
        raise ValueError('VLGuard configuration differs from supported native inference')
    index=read(source/'pytorch_model.bin.index.json')['weight_map']
    target=model.state_dict()
    loaded=set();derived_buffers=[];original_rotary=None
    def map_key(key):
        if key.startswith('model.mm_projector.0.'):
            return key.replace('model.mm_projector.0.','multi_modal_projector.linear_1.')
        if key.startswith('model.mm_projector.2.'):
            return key.replace('model.mm_projector.2.','multi_modal_projector.linear_2.')
        if key.startswith('model.vision_tower.'):
            return key.replace('model.vision_tower.vision_tower.','vision_tower.')
        if key.startswith('model.'):
            return 'language_model.'+key
        if key.startswith('lm_head.'):
            return 'language_model.'+key
        raise ValueError('Unexpected VLGuard source tensor '+key)
    for filename in sorted(set(index.values())):
        state=torch.load(source/filename,map_location='cpu',weights_only=True,mmap=True)
        mapped={}
        for key,value in state.items():
            if key.endswith('.self_attn.rotary_emb.inv_freq'):
                expected=1.0/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128))
                if not torch.allclose(value.float(),expected,rtol=5e-3,atol=1e-8):
                    raise ValueError('Original rotary frequencies differ from HF configuration')
                if original_rotary is not None and not torch.equal(original_rotary,value):
                    raise ValueError('Per-layer rotary buffers differ; cannot map to shared HF rotary')
                original_rotary=value.clone()
                derived_buffers.append(key)
                continue
            name_out=map_key(key)
            if name_out not in target:
                raise ValueError('Unexpected mapped tensor '+name_out)
            value=value.to(dtype=target[name_out].dtype)
            if name_out in ('language_model.model.embed_tokens.weight','language_model.lm_head.weight'):
                if value.shape[0]!=32000 or value.shape[1:]!=target[name_out].shape[1:]:
                    raise ValueError('Original vocab differs from verified 32000')
                expanded=torch.zeros_like(target[name_out]);expanded[:32000]=value;value=expanded
            if value.shape!=target[name_out].shape:
                raise ValueError('Shape mismatch '+name_out)
            mapped[name_out]=value
            loaded.add(name_out)
        result=model.load_state_dict(mapped,strict=False)
        if result.unexpected_keys:
            raise ValueError(result.unexpected_keys)
        # Every copied value must match the source in the inference dtype.
        if any(not torch.equal(model.state_dict()[k],v) for k,v in mapped.items()):
            raise ValueError('Converted VLGuard tensor mismatch')
        del state,mapped
    missing=set(target)-loaded
    if any(not k.startswith('vision_tower.') for k in missing):
        raise ValueError('Missing non-vision VLGuard weights '+str(sorted(missing)))
    if original_rotary is None or len(derived_buffers)!=32:
        raise ValueError('Missing source rotary buffers')
    # The released BF16 buffers include historical rounding (up to 0.423%).
    # Preserve their exact saved values instead of replacing with ideal FP32
    # frequencies. All 32 source layers share exactly the same buffer.
    rotary=model.language_model.model.rotary_emb
    rotary.inv_freq=original_rotary.float()
    rotary.original_inv_freq=original_rotary.float().clone()
    tokenizer=AutoTokenizer.from_pretrained(source,use_fast=False,local_files_only=True)
    if len(tokenizer)!=32000:
        raise ValueError('Unexpected source tokenizer vocabulary')
    tokenizer.add_special_tokens({'additional_special_tokens':['<image>'],'pad_token':'<pad>'})
    if tokenizer.convert_tokens_to_ids('<image>')!=32000 or tokenizer.pad_token_id!=32001:
        raise ValueError('Special token IDs differ from HF mapping')
    tokenizer.padding_side='left'
    processor.tokenizer=tokenizer
    audit=dict(source=str(source),loaded_tensors=len(loaded),frozen_vision_tensors=len(missing),
        all_copied_tensors_verified=True,original_vocabulary=32000,
        verified_derived_rotary_buffers=derived_buffers,
        appended_tokens='image/pad and alignment rows; suppressed from generation',
        frozen_vision_source=str(BASE),source_weights={f:sha(source/f) for f in sorted(set(index.values()))})
    audit['visual_preprocessing']='native pad to square with CLIP mean RGB background, then CLIP 336 resize'
    audit['prompt_tokenization']='original LLaVA split-at-image tokenizer; expand image marker to 576 CLIP patches'
    audit['vision_tensors_loaded_from_vlguard']=sum(k.startswith('vision_tower.') for k in loaded)
    audit['vision_tensors_inherited_from_base']=len(missing)
    audit['rotary_mapping']=dict(policy='preserve exact checkpoint frequencies in shared modern HF rotary',
        all_32_layers_identical=True,source_dtype=str(original_rotary.dtype),
        frequencies=original_rotary.float().tolist(),
        max_relative_deviation_from_ideal=float(((original_rotary.float()-expected).abs()/expected).max()))
    dump(audit_path,audit)
    return model,processor,audit

def inputs_for_vlguard(processor,records):
    """Match original LLaVA image_aspect_ratio=pad rather than center-cropping."""
    from PIL import Image,ImageOps
    images=[]
    for record in records:
        if record['modality']!='image':continue
        with Image.open(record['image_path']) as image:
            image=ImageOps.exif_transpose(image).convert('RGB')
        width,height=image.size
        if width!=height:
            side=max(width,height)
            square=Image.new('RGB',(side,side),tuple(int(x*255) for x in processor.image_processor.image_mean))
            square.paste(image,((side-width)//2,(side-height)//2));image=square
        images.append(image)
    tokenizer=processor.tokenizer
    token_sequences=[]
    for record in records:
        # Original tokenizer_image_token tokenizes chunks separately, including
        # the SentencePiece space after the image. HF's special-token expansion
        # otherwise removes that token and changes the native model's prompt.
        chunks=[tokenizer(s).input_ids for s in record['prompt'].split('<image>')]
        offset=int(bool(chunks[0]) and chunks[0][0]==tokenizer.bos_token_id)
        sequence=[tokenizer.bos_token_id] if offset else []
        for index,chunk in enumerate(chunks):
            if index:sequence.extend([32000]*576)
            sequence.extend(chunk[offset:])
        token_sequences.append(sequence)
    inputs=tokenizer.pad({'input_ids':token_sequences},padding=True,return_tensors='pt')
    if images:inputs.update(processor.image_processor(images=images,return_tensors='pt'))
    return inputs
