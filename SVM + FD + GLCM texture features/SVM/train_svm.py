
import argparse
import json
from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from feature_extraction import extract_dataset, FEATURE_NAMES, CLASSES
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from protocol import split_reactors, save_manifest, PROTOCOL, REACTORS
from dataset_paths import DATA_DIR,CSV_PATH,require_experiment_data

def metrics(df):
    a,b = df.true_stage,df.predicted_stage
    return dict(accuracy=accuracy_score(a,b),
                precision=precision_score(a,b,labels=CLASSES,average='macro',zero_division=0),
                recall=recall_score(a,b,labels=CLASSES,average='macro',zero_division=0),
                macro_f1=f1_score(a,b,labels=CLASSES,average='macro',zero_division=0))

def run_experiment(csv_path=CSV_PATH,image_dir=DATA_DIR,
                   output_dir=Path(__file__).resolve().parent/'results',num_repeats=5,base_seed=42):
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    require_experiment_data(pd.read_csv(csv_path,header=None))
    metadata,x=extract_dataset(csv_path,image_dir,32)
    if 'reactor_id' not in metadata:
        raise ValueError('Leave-one-reactor-out evaluation requires reactor_id in the CSV.')
    data=metadata.rename(columns={'image_name':'image','label':'stage'})
    data['reactor_id']=data.reactor_id.str.upper()
    summaries=[]
    for repeat in range(1,num_repeats+1):
        predictions=[]
        for fold,reactor in enumerate(REACTORS,1):
            seed=base_seed+100*(repeat-1)+(fold-1)
            train,val,test=split_reactors(data,reactor,seed)
            folder=output_dir/f'repeat_{repeat:02d}'/f'fold_{fold:02d}_test_{reactor}'
            folder.mkdir(parents=True,exist_ok=True)
            save_manifest(train,val,test,folder/'split_manifest.csv')
            model=Pipeline([('scaler',StandardScaler()),('svm',SVC(C=10,kernel='rbf',gamma='scale',probability=True,random_state=seed))])
            model.fit(x.loc[train.index],train.stage)
            result=test[['image','reactor_id','stage','was_used_for_training']].rename(columns={'stage':'true_stage'})
            result['predicted_stage']=model.predict(x.loc[test.index])
            result['repeat_id'],result['fold_id'],result['fold_seed']=repeat,fold,seed
            result.to_csv(folder/'final_test_predictions.csv',index=False,encoding='utf-8-sig')
            joblib.dump(dict(model=model,feature_names=FEATURE_NAMES,protocol=PROTOCOL,seed=seed,repeat_id=repeat,fold_id=fold,
                             train_images=train.image.tolist(),validation_images=val.image.tolist(),test_images=test.image.tolist(),
                             config={"glcm_levels":32},x_test=x.loc[test.index].reset_index(drop=True),y_test=test.stage.reset_index(drop=True)),folder/'svm_model.joblib')
            predictions.append(result)
        combined=pd.concat(predictions,ignore_index=True)
        if len(combined)!=len(data) or combined.image.duplicated().any() or combined.was_used_for_training.any():
            raise RuntimeError('Independent test aggregation validation failed.')
        folder=output_dir/f'repeat_{repeat:02d}'/'combined';folder.mkdir(exist_ok=True)
        combined.to_csv(folder/'combined_predictions_231.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(confusion_matrix(combined.true_stage,combined.predicted_stage,labels=CLASSES),index=CLASSES,columns=CLASSES).to_csv(folder/'confusion_matrix.csv')
        summaries.append(dict(repeat_id=repeat,**metrics(combined)))
    folder=output_dir/'five_repeats_summary';folder.mkdir(exist_ok=True)
    table=pd.DataFrame(summaries).set_index('repeat_id')
    table.to_csv(folder/'repeat_metrics.csv')
    pd.DataFrame({'mean':table.mean(),'std':table.std(ddof=1)}).to_csv(folder/'mean_and_std.csv')
    (output_dir/'experiment_config.json').write_text(json.dumps(dict(protocol=PROTOCOL,num_repeats=num_repeats,base_seed=base_seed,features=FEATURE_NAMES,std_ddof=0),indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description='SVM independent leave-one-reactor-out evaluation')
    p.add_argument('--csv-path',type=Path,default=CSV_PATH)
    p.add_argument('--image-dir',type=Path,default=DATA_DIR)
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    p.add_argument('--num-repeats',type=int,default=5)
    p.add_argument('--base-seed',type=int,default=42)
    a=p.parse_args()
    if a.num_repeats<1:p.error('The number of repetitions must be greater than 0')
    run_experiment(a.csv_path,a.image_dir,a.output_dir,a.num_repeats,a.base_seed)
if __name__=='__main__':main()
