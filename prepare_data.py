
from pathlib import Path
import sys
import argparse
import cv2
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'analysis'))
from fd import box_counting
sys.path.insert(0,str(ROOT))
from dataset_paths import DATA_DIR,SOURCE_CSV,CSV_PATH

def prepare(source,images,output):
    df=pd.read_csv(source,header=None,names=['image','fd','d50','feature3','stage','reactor_id'])
    for index,row in df.iterrows():
        # Convert color images to grayscale with OpenCV, following Text S2; retain the original image resolution.
        image=cv2.imdecode(np.fromfile(images/row.image,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError(f'Cannot read {row.image}')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        df.loc[index,'fd']=box_counting(gray)[0]
    output.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output,index=False,header=False,float_format='%.12g')
    print(f'Generated paper-method inputs: {len(df)} records: {output}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE_CSV)
    p.add_argument('--image-dir',type=Path,default=DATA_DIR)
    p.add_argument('--output',type=Path,default=CSV_PATH)
    args=p.parse_args()
    if args.output.resolve()==args.source.resolve():p.error('The output must differ from the source CSV.')
    prepare(args.source,args.image_dir,args.output)
if __name__=='__main__':main()
