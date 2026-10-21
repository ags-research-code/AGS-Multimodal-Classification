
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]

def predict(model,images,numeric,device,batch_size=12):
    values=[]
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            values.extend(model(images[start:start+batch_size].to(device),numeric[start:start+batch_size].to(device)).argmax(1).cpu().tolist())
    return np.array(values)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',choices=['ResNet18FD','ResNet34FD','ResNet50FD','ResNet101FD','ResNet152FD'],required=True)
    p.add_argument('--mode',choices=['permutation','correlation'],required=True)
    p.add_argument('--weights-dir',type=Path,default=None)
    p.add_argument('--training-results-dir',type=Path,default=None)
    p.add_argument('--output-dir',type=Path,default=None)
    a=p.parse_args();project=ROOT/a.project
    sys.path.insert(0,str(project));sys.path.insert(0,str(project/'src'))
    from data_loader import read_labeled_csv,MultiModalDataset
    from model import MultiModalModel
    from config import CSV_PATH,IMAGE_DIR,DEVICE
    from protocol import split_reactors
    weights=a.weights_dir or project/'models/saved';training_results=a.training_results_dir or project/'results'
    output=a.output_dir or ROOT/'analysis'/f'results_{a.mode}'/a.project
    output.mkdir(parents=True,exist_ok=True)
    df=read_labeled_csv(CSV_PATH);encoder=LabelEncoder().fit(df.stage)
    records=[]
    for repeat in range(1,6):
        base_correct=0;permuted_correct={k:np.zeros(30,dtype=int) for k in ('image','fd','d50')};correlations=[]
        for fold,reactor in enumerate(('Z1','Z2','Z3'),1):
            folder=f'fold_{fold:02d}_test_{reactor}';seed=42+100*(repeat-1)+fold-1
            train,val,test=split_reactors(df,reactor,seed)
            manifest=pd.read_csv(training_results/f'repeat_{repeat:02d}'/folder/'split_manifest.csv')
            for data,name in ((train,'train'),(val,'validation'),(test,'test')):
                if set(data.image)!=set(manifest.loc[manifest.subset.eq(name),'image']):raise ValueError('The weight-associated split does not match the independent test protocol in the paper.')
            path=weights/f'repeat_{repeat:02d}'/folder/'final_model.pth'
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if saved.get('protocol')!='loro_independent_80_20_v1' or saved.get('seed')!=seed:raise ValueError('The weights lack the independent leave-one-reactor-out protocol or have an inconsistent random seed.')
            if saved['classes']!=list(encoder.classes_):raise ValueError('The class order in the weights does not match.')
            model=MultiModalModel(3,pretrained=False).to(DEVICE);model.load_state_dict(saved['model_state_dict']);model.eval()
            ds=MultiModalDataset(test,image_dir=IMAGE_DIR,label_encoder=encoder)
            samples=[ds[i] for i in range(len(ds))];images=torch.stack([v['images'] for v in samples]);numeric=torch.stack([v['data'] for v in samples]);truth=encoder.transform(test.stage)
            if a.mode=='permutation':
                base_correct+=int((predict(model,images,numeric,DEVICE)==truth).sum())
                for ki,key in enumerate(permuted_correct):
                    for permutation in range(30):
                        order=np.random.default_rng(seed+10000*(ki+1)+permutation).permutation(len(test))
                        img,num=images,numeric
                        if key=='image':img=images[order]
                        else:
                            num=numeric.clone();col={'fd':0,'d50':1}[key];num[:,col]=numeric[order,col]
                        permuted_correct[key][permutation]+=int((predict(model,img,num,DEVICE)==truth).sum())
            else:
                features=[]
                with torch.inference_mode():
                    for start in range(0,len(images),12):features.append(model.image_model(images[start:start+12].to(DEVICE)).cpu().numpy())
                matrix=np.concatenate(features);valid=[]
                for k in range(matrix.shape[1]):
                    if np.ptp(matrix[:,k])==0:continue
                    rho=spearmanr(matrix[:,k],test.fd.to_numpy()).statistic
                    if np.isfinite(rho):valid.append(abs(float(rho)))
                if not valid:raise ValueError('This fold has no valid image-channel correlation coefficients.')
                median=float(np.median(valid));correlations.append(median)
                records.append(dict(repeat=repeat,fold=fold,test_reactor=reactor,median_abs_rho=median,valid_channels=len(valid)))
            del model,ds,samples
            if torch.cuda.is_available():torch.cuda.empty_cache()
        if a.mode=='permutation':
            for key,correct in permuted_correct.items():
                for permutation,value in enumerate(correct,1):records.append(dict(repeat=repeat,input=key,permutation=permutation,baseline_accuracy=base_correct/len(df),permuted_accuracy=int(value)/len(df),importance=(base_correct-int(value))/len(df)))
    table=pd.DataFrame(records);table.to_csv(output/'raw.csv',index=False)
    if a.mode=='permutation':means=table.groupby(['repeat','input']).importance.mean().unstack('input')
    else:means=table.groupby('repeat').median_abs_rho.mean().to_frame('median_abs_rho')
    means.to_csv(output/'repeat_means.csv');pd.DataFrame({'mean':means.mean(),'sd':means.std(ddof=1)}).to_csv(output/'mean_and_sd.csv')
if __name__=='__main__':main()
