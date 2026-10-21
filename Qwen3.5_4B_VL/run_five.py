
from pathlib import Path
import argparse
import subprocess
import sys
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from dataset_paths import CSV_PATH,IMAGE_DIR
CLASSES=['growth','stabilization','disintegration']
sys.path.insert(0,str(HERE/'src'))
from data_loader import choose_few_shot_examples
def summarize(output_dir, csv_path=CSV_PATH):
    output_dir=Path(output_dir)
    source=pd.read_csv(csv_path,header=None,dtype=str,keep_default_na=False)
    if str(source.iloc[0,0]).strip().lower() in {'image','image_name','filename','file_name'}:
        source=source.iloc[1:].copy()
    if source.shape[1]!=6:raise ValueError('The input CSV must have six columns.')
    source.columns=['image','fd','d50','feature3','stage','reactor_id']
    source['reactor_id']=source.reactor_id.str.strip().str.upper()
    if source.empty or source.image.duplicated().any() or not source.stage.isin(CLASSES).all():raise ValueError('The input CSV is empty, contains duplicates, or has invalid classes.')
    expected=dict(zip(source.image,source.stage))
    records=[]
    for k in range(1,6):
        p=output_dir/f'config_{k:02d}'/'predictions.csv'
        df=pd.read_csv(p)
        if len(df)!=len(expected) or df.image.duplicated().any() or set(df.image)!=set(expected):
            raise ValueError(f'{p} does not fully cover the {len(expected)} images in the input CSV.')
        if not df.true_stage.eq(df.image.map(expected)).all():
            raise ValueError(f'{p} has labels that differ from the source data.')
        if not df.status.eq('ok').all() or not df.predicted_stage.isin(CLASSES).all():
            raise ValueError(f'{p} contains failed or invalid predictions; retry before aggregation.')
        example_config=HERE/'examples'/f'config_{k:02d}.json'
        expected_examples={name:'|'.join(row['image'] for row in choose_few_shot_examples(source,query_image=name,example_config=example_config)) for name in source.image}
        if 'few_shot_images' not in df or not df.few_shot_images.eq(df.image.map(expected_examples)).all():
            raise ValueError(f'{p} uses examples that do not match configuration {k}.')
        a,b=df.true_stage,df.predicted_stage
        records.append(dict(configuration=k,accuracy=accuracy_score(a,b),
            precision=precision_score(a,b,labels=CLASSES,average='macro',zero_division=0),
            recall=recall_score(a,b,labels=CLASSES,average='macro',zero_division=0),
            macro_f1=f1_score(a,b,labels=CLASSES,average='macro',zero_division=0)))
    table=pd.DataFrame(records).set_index('configuration')
    table.to_csv(output_dir/'configuration_metrics.csv')
    pd.DataFrame({'mean':table.mean(),'std':table.std(ddof=1)}).to_csv(output_dir/'mean_and_std.csv')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=HERE/'results/five_configurations')
    p.add_argument('--csv-path',type=Path,default=CSV_PATH)
    p.add_argument('--image-dir',type=Path,default=IMAGE_DIR)
    p.add_argument('--summarize-only',action='store_true')
    args,extra=p.parse_known_args()
    if not args.summarize_only:
        from data_loader import read_dataset
        source=read_dataset(args.csv_path,args.image_dir)
        if source.stage.value_counts().min()<4:
            p.error('The 10 public examples are insufficient for per-image nine-shot evaluation; each class needs at least 4 images. Use the complete local dataset.')
        for k in range(1,6):
            subprocess.run([sys.executable,'-B',str(HERE/'src/predict.py'),
                '--example-config',str(HERE/'examples'/f'config_{k:02d}.json'),
                '--csv-path',str(args.csv_path),
                '--image-dir',str(args.image_dir),
                '--output-dir',str(args.output_dir/f'config_{k:02d}'),*extra],check=True)
    summarize(args.output_dir,args.csv_path)
if __name__=='__main__':main()
