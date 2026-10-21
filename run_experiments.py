
import argparse
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
PROJECTS=['SVM','UNM512','EfficientNet_B0','ViT_S/16']+[f'ResNet{n}{suffix}' for suffix in ('','FD') for n in (18,34,50,101,152)]+['ResNet34FDGated']
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',default=['all'])
    p.add_argument('--num-repeats',type=int,default=5)
    p.add_argument('--paired',action='store_true',help='Run ResNet34, ResNet34FD, and ResNet34FDGated 20 times each and save separately')
    a=p.parse_args()
    models=['ResNet34','ResNet34FD','ResNet34FDGated'] if a.paired else (PROJECTS if a.models==['all'] else a.models)
    if any(x not in PROJECTS for x in models):p.error('Unknown model; available: '+', '.join(PROJECTS))
    if a.num_repeats<1:p.error('The number of repetitions must be greater than 0')
    import pandas as pd
    from dataset_paths import CSV_PATH,require_experiment_data
    require_experiment_data(pd.read_csv(CSV_PATH,header=None))
    for name in models:
        project=ROOT/('SVM + FD + GLCM texture features' if name=='SVM' else name)
        command=[sys.executable,'-B',str(project/'src/train.py'),'--num-repeats',str(20 if a.paired else a.num_repeats)]
        if a.paired:command+=['--results-dir',str(project/'results_20'),'--model-dir',str(project/'models/saved_20')]
        subprocess.run(command,cwd=project,check=True)
if __name__=='__main__':main()
