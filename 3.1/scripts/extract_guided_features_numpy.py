"""Bounded-memory FP32 CLIP ViT extraction for CPU-only diagnostics.

No torch/CUDA libraries or full-model allocation. Safetensors weights are
memory mapped per operation. GPU training uses the original Transformers tower.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from PIL import Image, ImageOps


class TensorReader:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            self.header = json.loads(stream.read(length))
        self.start = 8 + length

    def get(self, name, rows=None):
        info = self.header[name]
        dtype = {'F32':'<f4', 'F16':'<f2', 'BF16':'<u2'}[info['dtype']]
        array = np.memmap(self.path, dtype=dtype, mode='r', shape=tuple(info['shape']),
                          offset=self.start + info['data_offsets'][0])
        if rows is not None:
            array = array[rows]
        if info['dtype'] == 'BF16':
            return (array.astype(np.uint32) << 16).view(np.float32)
        return np.asarray(array, dtype=np.float32)


def linear(x, reader, prefix):
    return x @ reader.get(prefix+'.weight').T + reader.get(prefix+'.bias')


def layer_norm(x, reader, prefix, eps=1e-5):
    mean=x.mean(-1,keepdims=True)
    variance=((x-mean)**2).mean(-1,keepdims=True)
    return (x-mean)/np.sqrt(variance+eps)*reader.get(prefix+'.weight')+reader.get(prefix+'.bias')


def vision_features(pixels, reader, *, hidden_size=768, heads=12, patch_size=16,
                    selected_blocks=11, eps=1e-5):
    """FP32 CLIP pre-LN transformer, default selected output without CLS."""
    _,height,width=pixels.shape
    grid_h,grid_w=height//patch_size,width//patch_size
    patches=pixels.reshape(3,grid_h,patch_size,grid_w,patch_size).transpose(1,3,0,2,4).reshape(grid_h*grid_w,-1)
    weight=reader.get('vision_model.embeddings.patch_embedding.weight').reshape(hidden_size,-1)
    hidden=patches@weight.T
    hidden=np.concatenate((reader.get('vision_model.embeddings.class_embedding')[None],hidden),0)
    hidden=hidden+reader.get('vision_model.embeddings.position_embedding.weight')
    hidden=layer_norm(hidden,reader,'vision_model.pre_layrnorm',eps)
    count=hidden.shape[0]
    width_head=hidden_size//heads
    for layer in range(selected_blocks):
        prefix=f'vision_model.encoder.layers.{layer}'
        x=layer_norm(hidden,reader,prefix+'.layer_norm1',eps)
        q,k,v=[linear(x,reader,prefix+'.self_attn.'+name).reshape(count,heads,width_head).transpose(1,0,2)
               for name in ('q_proj','k_proj','v_proj')]
        scores=(q * np.float32(width_head**-0.5))@k.transpose(0,2,1)
        scores-=scores.max(-1,keepdims=True)
        probabilities=np.exp(scores)
        probabilities/=probabilities.sum(-1,keepdims=True)
        attention=(probabilities@v).transpose(1,0,2).reshape(count,hidden_size)
        hidden=hidden+linear(attention,reader,prefix+'.self_attn.out_proj')
        x=linear(layer_norm(hidden,reader,prefix+'.layer_norm2',eps),reader,prefix+'.mlp.fc1')
        # Original CLIP uses QuickGELU, not the adapter's exact GELU.
        x=x/(1+np.exp(np.clip(-1.702*x,-80,80)))
        hidden=hidden+linear(x,reader,prefix+'.mlp.fc2')
    return hidden[1:].copy()


def preprocess(path, config):
    with Image.open(path) as opened:
        image=ImageOps.exif_transpose(opened).convert('RGB')
    short=config['size']['shortest_edge']
    width,height=image.size
    # Transformers shortest-edge resize uses integer truncation.
    if width<height:
        size=(short,int(short*height/width))
    else:
        size=(int(short*width/height),short)
    image=image.resize(size,resample=config.get('resample',3))
    crop=config['crop_size']
    left=(size[0]-crop['width'])//2
    top=(size[1]-crop['height'])//2
    image=image.crop((left,top,left+crop['width'],top+crop['height']))
    array=np.asarray(image,dtype=np.float32)*np.float32(config.get('rescale_factor',1/255))
    array=(array-np.asarray(config['image_mean'],dtype=np.float32))/np.asarray(config['image_std'],dtype=np.float32)
    return array.transpose(2,0,1).copy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--moralclip-model',type=Path,required=True)
    parser.add_argument('--base-model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from tokenizers import Tokenizer
    selected=json.loads(args.samples.read_text())
    vision=TensorReader(args.moralclip_model/'model.safetensors')
    config=json.loads((args.moralclip_model/'preprocessor_config.json').read_text())
    tokenizer=Tokenizer.from_file(str(args.base_model/'tokenizer.json'))
    index=json.loads((args.base_model/'model.safetensors.index.json').read_text())['weight_map']
    key='language_model.model.embed_tokens.weight'
    embedding=TensorReader(args.base_model/index[key])
    values={}
    ids_log={}
    for row in selected['train']:
        if row['task'] in ids_log:continue
        human=row['prompt'][len('USER: '):-len(' ASSISTANT:')].replace('<image>','').strip()
        ids=tokenizer.encode(human,add_special_tokens=False).ids
        ids_log[row['task']]=ids
        values['condition_'+row['task']]=embedding.get(key,rows=ids).mean(0)
    all_rows=selected['train']+selected['validation']
    paths=list(dict.fromkeys(r['image_path'] for r in all_rows))
    parts=args.output.parent/'feature-parts'
    parts.mkdir(exist_ok=True)
    for i,path in enumerate(paths):
        cached=parts/f'image-{i}.npy'
        if cached.exists():
            values['image_'+str(i)]=np.load(cached)
        else:
            values['image_'+str(i)]=vision_features(preprocess(path,config),vision)
            np.save(cached,values['image_'+str(i)])
        print(f'NumPy real MoralCLIP image {i+1}/{len(paths)}',flush=True)
    np.savez(args.output,**values)
    args.output.with_suffix('.json').write_text(json.dumps({'paths':paths,'condition_token_ids':ids_log},indent=2)+'\n')


if __name__=='__main__':
    main()
