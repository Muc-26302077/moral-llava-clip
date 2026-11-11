"""Result uncertainty is paired by scene; incomplete runs cannot fill prose."""
import json
from pathlib import Path
import tempfile
import unittest

from moral_llava import polarity
from moral_llava.guided_report import build_report, paired_intervals


class GuidedReportTests(unittest.TestCase):
    def records(self):
        data=[]
        before=[]
        after=[]
        for scene in ('a','b'):
            for task in polarity.TASKS:
                for modality in ('image','text'):
                    row={'id':f'{scene}::{task}::{modality}','scene_id':scene,'task':task,'modality':modality,
                         'conversations':[{}, {'value':polarity.ANSWERS[task][True]}]}
                    data.append(row)
                    wrong=polarity.ANSWERS[task][False]
                    right=polarity.ANSWERS[task][True]
                    before.append(dict(row,raw_prediction=wrong if modality=='image' else right))
                    after.append(dict(row,raw_prediction=right))
        return data,before,after

    def test_identical_predictions_have_zero_paired_difference(self):
        data,before,_=self.records()
        intervals=paired_intervals(data,before,before,draws=100)
        self.assertEqual(intervals,{'ticr':[0,0],'joint_accuracy':[0,0],'image_accuracy':[0,0]})

    def test_perfect_paired_improvement(self):
        data,before,after=self.records()
        intervals=paired_intervals(data,before,after,draws=100)
        self.assertEqual(intervals,{'ticr':[1,1],'joint_accuracy':[1,1],'image_accuracy':[1,1]})

    def test_smoke_cannot_generate_formal_outcome_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'frozen_config.json').write_text(json.dumps({'arguments':{'command':'smoke'}}))
            with self.assertRaisesRegex(ValueError,'cannot populate'):
                build_report(path,[])
            self.assertFalse((path/'成果总结.md').exists())


if __name__=='__main__':
    unittest.main()
