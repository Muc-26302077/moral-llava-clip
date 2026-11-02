"""Numerical and causal-position checks for paired continuation training."""
import unittest
from pathlib import Path

from moral_llava.train_crossmodal import (BASE, ROOT, PairCollator, load_splits,
                                          paired_state_loss, state_logits)


class PairTrainingTests(unittest.TestCase):
    def test_js_zero_for_identical_distributions_and_symmetric(self):
        import torch
        torch.manual_seed(3)
        a, b = torch.randn(2,5,3), torch.randn(2,5,3)
        classes = torch.randint(0,3,(2,5))
        _, equal = paired_state_loss(a,a,classes)
        _, ab = paired_state_loss(a,b,classes)
        _, ba = paired_state_loss(b,a,classes)
        self.assertAlmostEqual(float(equal),0,places=6)
        self.assertGreater(float(ab),0)
        self.assertAlmostEqual(float(ab),float(ba),places=6)

    def test_both_modalities_receive_finite_gradients(self):
        import torch
        a = torch.randn(2,5,3,requires_grad=True)
        b = torch.randn(2,5,3,requires_grad=True)
        ce, js = paired_state_loss(a,b,torch.zeros((2,5),dtype=torch.long))
        (ce+js).backward()
        for tensor in (a,b):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()),0)

    def test_gather_uses_requested_batch_positions_and_candidate_ids(self):
        import torch
        logits=torch.arange(2*9*11).reshape(2,9,11).float()
        positions=torch.tensor([[0,1,2,3,4],[4,5,6,7,8]])
        candidates=torch.tensor([2,5,8])
        result=state_logits(logits,positions,candidates)
        for batch in range(2):
            for dim in range(5):
                self.assertEqual(result[batch,dim].tolist(),logits[batch,positions[batch,dim],candidates].tolist())

    def test_real_collator_masks_prompts_and_aligns_state_prediction_positions(self):
        from transformers import AutoProcessor
        from types import SimpleNamespace
        processor=AutoProcessor.from_pretrained(str(BASE),local_files_only=True)
        from moral_llava.train import configure_processor
        from moral_llava.crossmodal import read_json
        config=SimpleNamespace(**read_json(BASE/'config.json'))
        config.vision_config=SimpleNamespace(**config.vision_config)
        expected=configure_processor(processor,config,padding_side='right')
        splits=load_splits(ROOT/'data/raw/SMID_moralclip')
        collator=PairCollator(processor,expected)
        batch=collator(splits['train'][:2])
        for modality in ('image','text'):
            for row in range(2):
                positions=batch[modality+'_positions'][row]
                ids=batch[modality]['input_ids'][row]
                labels=batch[modality]['labels'][row]
                for dim,pos in enumerate(positions):
                    cls=batch['state_classes'][row,dim]
                    self.assertEqual(int(ids[pos+1]),int(batch['candidate_ids'][cls]))
                    self.assertEqual(int(labels[pos+1]),int(ids[pos+1]))
                active=(labels!=-100).nonzero().flatten()
                self.assertTrue((labels[:active[0]]==-100).all())
                self.assertEqual(int(labels[active[-1]]),processor.tokenizer.eos_token_id)
        self.assertNotIn('pixel_values',batch['text'])


if __name__=='__main__':unittest.main()
