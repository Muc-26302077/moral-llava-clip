import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from moralise import parse_letter,parse_multi,score_moralise

class ScoringTests(unittest.TestCase):
    def test_strict_parsing_does_not_extract_letters_from_prose(self):
        self.assertEqual(parse_letter('(B)','AB'),'B')
        self.assertIsNone(parse_letter('Because this causes harm.','AB'))
        self.assertIsNone(parse_letter('AB','AB'))
        self.assertEqual(parse_multi('(A), (C)'),{'justice','harm'})
        self.assertIsNone(parse_multi('A and C'))
        self.assertIsNone(parse_multi('N, C'))
        self.assertEqual(parse_multi('N'),set())

    def fixture(self):
        records=[];predictions=[]
        for task in ['S1','S2','S3']:
            for n,category in enumerate(['care','harm']):
                records.append(dict(id=f'{task}:{n}',task=task,category=category,
                    gold_morality='B',gold_topics=[category]))
                answer={'S1':'B','S2':'F' if category=='care' else 'C',
                        'S3':'F' if category=='care' else 'C'}[task]
                predictions.append(dict(id=f'{task}:{n}',raw_prediction=answer))
        return records,predictions

    def test_missing_duplicate_predictions_rejected(self):
        refs,preds=self.fixture()
        with self.assertRaises(ValueError):score_moralise(refs,preds[:-1])
        with self.assertRaises(ValueError):score_moralise(refs,preds+[preds[0]])

    def test_invalid_outputs_stay_in_denominator(self):
        refs,preds=self.fixture()
        preds[0]['raw_prediction']='I refuse.'
        result=score_moralise(refs,preds)
        self.assertEqual(result['tasks']['S1']['questions'],2)
        self.assertEqual(result['tasks']['S1']['valid_output_rate'],.5)
        self.assertEqual(result['tasks']['S1']['category_macro_score'],.5)

    def test_joint_input_task_is_not_reported_as_ticr(self):
        refs,preds=self.fixture()
        result=score_moralise(refs,preds)
        self.assertEqual(result['protocol'],'moralise-native-v1')
        self.assertNotIn('ticr',result)
        self.assertEqual(result['tasks']['S2']['micro_score'],1)

if __name__=='__main__':unittest.main()
