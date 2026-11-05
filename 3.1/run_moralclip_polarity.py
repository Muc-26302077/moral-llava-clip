"""Original MoralCLIP + LLaVA conditional polarity training and inference."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ('train','infer','score','majority','prepare'):
        print('Usage: run_moralclip_polarity.py {prepare,train,infer,score,majority} [options]')
        return 0 if argv in ([],['-h'],['--help']) else 2
    command,extra=argv[0],argv[1:]
    if command in ('prepare','score','majority'):
        from run_polarity import main as polarity
        return polarity(argv)
    data=ROOT/'data/polarity-v1'
    defaults=['--image-root',str(ROOT/'data/raw/SMID_moralclip'),'--task','polarity']
    if command=='train':
        from moral_llava.train import main as train
        return train(defaults+['--train-data',str(data/'train.json'),
                     '--validation-data',str(data/'validation.json'),
                     '--moralclip-model','anaaa2/moralclip-base',
                     '--revision','b234b804b114d9e37bb655e11cbbb5f5e971b7a9']+extra)
    from moral_llava.infer import main as infer
    return infer(defaults+['--data',str(data/'test.json'),'--max-new-tokens','16']+extra)


if __name__=='__main__':
    raise SystemExit(main())
