
import argparse
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score,f1_score
sys.path.insert(0,str(Path(__file__).resolve().parent))
from fd import holm
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from dataset_paths import DATA_DIR,CSV_PATH,require_experiment_data
CLASSES=['growth','stabilization','disintegration']

def exact_signed_rank(differences):
    d=np.round(np.asarray(differences,dtype=float),12)
    if not np.isfinite(d).all():raise ValueError('Paired differences contain NaN or infinity.')
    d=d[d!=0]
    if not len(d):return dict(n_effective=0,W=0.,p_raw=1.)
    ranks2=np.rint(2*rankdata(np.abs(d),method='average')).astype(int)
    positive=int(ranks2[d>0].sum());total=int(ranks2.sum());w=min(positive,total-positive)
    # Use dynamic programming to count all sign assignments exactly, handling tied ranks and zero differences correctly.
    counts=[1]+[0]*total;reachable=0
    for rank in ranks2:
        for s in range(reachable,-1,-1):counts[s+rank]+=counts[s]
        reachable+=rank
    extreme=sum(c for s,c in enumerate(counts) if min(s,total-s)<=w)
    return dict(n_effective=len(d),W=w/2,p_raw=extreme/(2**len(d)))

def load_runs(directory):
    source=pd.read_csv(CSV_PATH,header=None);expected=dict(zip(source[0],source[4]));expected_reactors=dict(zip(source[0],source[5]));runs=[]
    require_experiment_data(source)
    for repeat in range(1,21):
        folder=directory/f'repeat_{repeat:02d}'
        df=pd.read_csv(folder/'combined/combined_predictions_231.csv')
        if len(df)!=231 or df.image.duplicated().any() or set(df.image)!=set(expected) or not df.true_stage.eq(df.image.map(expected)).all():
            raise ValueError(f'{folder} is missing complete, valid per-image records for 231 images.')
        if not df.predicted_stage.isin(CLASSES).all() or not df.reactor_id.eq(df.image.map(expected_reactors)).all():raise ValueError('Predicted classes or sample reactor identifiers are invalid.')
        manifests=[]
        for fold,reactor in enumerate(('Z1','Z2','Z3'),1):
            m=pd.read_csv(folder/f'fold_{fold:02d}_test_{reactor}'/'split_manifest.csv')
            if len(m)!=231 or m.image.duplicated().any() or set(m.image)!=set(expected):raise ValueError('The split manifest is incomplete.')
            if not m.reactor_id.eq(m.image.map(expected_reactors)).all():raise ValueError('Reactor identifiers in the split manifest do not match the source data.')
            if set(m.subset)!=set(['train','validation','test']):raise ValueError('Split labels are invalid.')
            if not m.loc[m.subset.eq('test'),'reactor_id'].eq(reactor).all() or m.loc[~m.subset.eq('test'),'reactor_id'].eq(reactor).any():raise ValueError('Test reactor leakage detected.')
            part=df.loc[df.reactor_id.eq(reactor)]
            if set(part.image)!=set(m.loc[m.subset.eq('test'),'image']):raise ValueError('Prediction samples do not match the test manifest for this fold.')
            if not part.fold_seed.eq(42+100*(repeat-1)+fold-1).all():raise ValueError('Random seeds do not match across paired models.')
            manifests.append(m.sort_values('image')[['image','subset']].to_json(orient='records'))
        runs.append(dict(repeat=repeat,accuracy=accuracy_score(df.true_stage,df.predicted_stage),macro_f1=f1_score(df.true_stage,df.predicted_stage,labels=CLASSES,average='macro',zero_division=0),manifests=manifests))
    return runs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resnet',type=Path,required=True);p.add_argument('--fd',type=Path,required=True);p.add_argument('--gated',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'analysis/results_paired/paired_tests.json')
    args=p.parse_args();models=[load_runs(x) for x in (args.resnet,args.fd,args.gated)]
    if any(models[0][i]['manifests']!=other[i]['manifests'] for other in models[1:] for i in range(20)):raise ValueError('Training/validation/test splits differ across the three models.')
    results=[]
    for metric in ('macro_f1','accuracy'):
        family=[]
        for first,second,label in ((0,1,'ResNet34FD - ResNet34'),(1,2,'ResNet34FD-Gated - ResNet34FD')):
            x=np.array([v[metric] for v in models[first]]);y=np.array([v[metric] for v in models[second]])
            family.append(dict(comparison=label,metric=metric,model1_mean=x.mean(),model1_sd=x.std(ddof=1),model2_mean=y.mean(),model2_sd=y.std(ddof=1),mean_difference=(y-x).mean(),improved=int((y>x).sum()),unchanged=int((y==x).sum()),decreased=int((y<x).sum()),**exact_signed_rank(y-x)))
        for item,adj in zip(family,holm([v['p_raw'] for v in family])):item['p_holm']=float(adj)
        results.extend(family)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(results,indent=2),encoding='utf-8')
if __name__=='__main__':main()
