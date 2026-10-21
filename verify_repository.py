
from pathlib import Path
import argparse
import ast
import json
import hashlib
import sys
import pandas as pd
from protocol import split_reactors
from dataset_paths import DATA_DIR,CSV_PATH,SOURCE_CSV,EXAMPLE_DIR,require_experiment_data
ROOT=Path(__file__).resolve().parent
COLUMNS=['image','fd','d50','feature3','stage','reactor_id']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    for path in ROOT.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    directory=DATA_DIR if args.full else EXAMPLE_DIR
    raw=pd.read_csv(directory/'data_paper_method.csv',header=None,names=COLUMNS)
    source=pd.read_csv(directory/'data.csv',header=None,names=COLUMNS)
    if raw.empty or raw.image.duplicated().any():raise ValueError('The input is empty or contains duplicate images.')
    if not raw.drop(columns='fd').equals(source.drop(columns='fd')):raise ValueError('Image names, labels, or other numerical values differ between the two CSV files.')
    if set(raw.stage)!={'growth','stabilization','disintegration'} or set(raw.reactor_id)!={'Z1','Z2','Z3'}:raise ValueError('Classes or reactors are incomplete.')
    if raw[['fd','d50','feature3']].isna().any().any():raise ValueError('Numerical columns contain missing values.')
    if not all((directory/name).is_file() for name in raw.image):raise ValueError('Images listed in the CSV are missing.')
    if not args.full:
        if len(raw)!=10:raise ValueError('The public examples must contain exactly 10 records.')
        images={p.name for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg'}}
        if images!=set(raw.image):raise ValueError('Repository images do not match the 10 examples.')
        for path in ROOT.rglob('*.csv'):
            if len(pd.read_csv(path,header=None))>10:raise ValueError(f'A CSV with more than 10 rows remains in the public directory: {path}')
        info=json.loads((directory/'dataset_info.json').read_text(encoding='utf-8'))
        for item in info['images']:
            if hashlib.sha256((directory/item['image']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('The example image checksum does not match.')
    else:
        require_experiment_data(raw)
        for repeat in range(1,21):
            for fold,reactor in enumerate(('Z1','Z2','Z3'),1):split_reactors(raw,reactor,42+100*(repeat-1)+fold-1)
    count=0
    for path in ROOT.rglob('config.py'):
        values={}
        for node in ast.parse(path.read_text(encoding='utf-8-sig')).body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
                try:values[node.targets[0].id]=ast.literal_eval(node.value)
                except (ValueError,TypeError):pass
        if 'NUM_EPOCHS' in values:
            if not(values['NUM_EPOCHS']==100 and values['BASE_SEED']==42 and values['BATCH_SIZE']==12 and values['VALIDATION_FRACTION']==.2):raise ValueError('Training parameters do not match the method settings.')
            count+=1
    if count!=14:raise ValueError('The number of neural network modules does not match.')
    if args.full:
        sys.path.insert(0,str(ROOT/'Qwen3.5_4B_VL'));sys.path.insert(0,str(ROOT/'Qwen3.5_4B_VL/src'))
        from data_loader import choose_few_shot_examples
        for path in sorted((ROOT/'Qwen3.5_4B_VL/examples').glob('*.json')):
            for query in raw.image:
                choices=choose_few_shot_examples(raw,query,example_config=path)
                if len(choices)!=9 or query in [x['image'] for x in choices]:raise ValueError('The Qwen example-exclusion check failed.')
    print('Passed: '+('complete dataset and 60 independent splits' if args.full else '10 example images, two 10-row CSV files, image hashes, and public data scope')+'; parameters of 14 models and syntax of all programs.')
if __name__=='__main__':main()
