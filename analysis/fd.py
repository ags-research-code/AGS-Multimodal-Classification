
import argparse
from pathlib import Path
import json
import itertools
import cv2
import numpy as np
import pandas as pd
from scipy.stats import kruskal,rankdata,norm
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from dataset_paths import DATA_DIR,CSV_PATH,require_experiment_data

def box_counting(gray):
    threshold,_=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    foreground=gray<=threshold
    scales=np.array([5,10,20,40,80,160,320])
    counts=[]
    for size in scales:
        blocks=np.add.reduceat(np.add.reduceat(foreground,np.arange(0,len(gray),size),axis=0),np.arange(0,gray.shape[1],size),axis=1)
        counts.append(int((blocks>0).sum()))
    counts=np.array(counts)
    if (counts[2:5]==0).any():raise ValueError('No foreground exists at the fitting scales.')
    fd,intercept=np.polyfit(np.log(max(gray.shape)/scales[2:5]),np.log(counts[2:5]),1)
    return float(fd),float(threshold),counts.tolist()

def holm(p_values):
    order=np.argsort(p_values);out=np.empty(len(order));previous=0.
    for i,k in enumerate(order):
        previous=max(previous,min(1.,p_values[k]*(len(order)-i)));out[k]=previous
    return out

def stage_statistics(df):
    groups=[df.loc[df.stage.eq(x),'fd'].to_numpy() for x in ('growth','stabilization','disintegration')]
    h,p=kruskal(*groups)
    rank=rankdata(df.fd);n=len(df);_,counts=np.unique(df.fd,return_counts=True)
    variance=n*(n+1)/12-np.sum(counts**3-counts)/(12*(n-1))
    rank_means={x:rank[df.stage.eq(x)].mean() for x in df.stage.unique()}
    sizes=df.stage.value_counts();pairs=[]
    pair_list = itertools.combinations(('growth','stabilization','disintegration'),2) if p < 0.05 else []
    for x,y in pair_list:
        z=(rank_means[x]-rank_means[y])/np.sqrt(variance*(1/sizes[x]+1/sizes[y]))
        pairs.append(dict(stage1=x,stage2=y,z=float(z),p_raw=float(2*norm.sf(abs(z)))))
    for item,adj in zip(pairs,holm([x['p_raw'] for x in pairs])):item['p_holm']=float(adj)
    summary=[]
    for stage,g in df.groupby('stage'):
        x=g.fd
        summary.append(dict(stage=stage,n=len(x),mean=x.mean(),sd=x.std(ddof=1),median=x.median(),iqr=x.quantile(.75)-x.quantile(.25)))
    return dict(summary=summary,kruskal_h=float(h),kruskal_p=float(p),dunn_holm=pairs)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,default=CSV_PATH)
    p.add_argument('--image-dir',type=Path,default=DATA_DIR)
    p.add_argument('--output-dir',type=Path,default=ROOT/'analysis/results_fd')
    p.add_argument('--recalculate',action='store_true',help='Recalculate FD from the original images without overwriting the source CSV')
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.csv,header=None,names=['image','fd','d50','feature3','stage','reactor_id'])
    if args.recalculate:
        values=[]
        for row in df.itertuples():
            image=cv2.imdecode(np.fromfile(args.image_dir/row.image,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError(f'Cannot read {row.image}')
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            fd,t,counts=box_counting(gray);values.append(dict(image=row.image,fd_recalculated=fd,otsu_threshold=t,box_counts=json.dumps(counts)))
        recalculated=pd.DataFrame(values)
        recalculated.to_csv(args.output_dir/'fd_recalculated.csv',index=False)
        df['fd']=df.image.map(recalculated.set_index('image').fd_recalculated)
    (args.output_dir/'stage_statistics.json').write_text(json.dumps(stage_statistics(df),indent=2),encoding='utf-8')
if __name__=='__main__':main()
