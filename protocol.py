
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset_paths import require_experiment_data

PROTOCOL = 'loro_independent_80_20_v1'
REACTORS = ('Z1', 'Z2', 'Z3')

def split_reactors(df, test_reactor, seed, validation_fraction=0.2):
    require_experiment_data(df)
    if test_reactor not in REACTORS:
        raise ValueError('The test reactor must be Z1/Z2/Z3.')
    if set(df['reactor_id']) != set(REACTORS):
        raise ValueError('The data must contain exactly Z1/Z2/Z3.')
    if df['image'].duplicated().any():
        raise ValueError('Duplicate image filenames are not allowed.')
    pool = df.loc[~df['reactor_id'].eq(test_reactor)].copy()
    test = df.loc[df['reactor_id'].eq(test_reactor)].copy()
    # The paper specifies a random 80:20 split without stratification, so no additional stratification rule is introduced.
    train, val = train_test_split(pool, test_size=validation_fraction,
                                  random_state=int(seed), shuffle=True)
    train, val, test = [x.sort_index().copy() for x in (train, val, test)]
    groups = [set(x['image']) for x in (train, val, test)]
    if any(groups[i] & groups[j] for i,j in ((0,1),(0,2),(1,2))):
        raise RuntimeError('Training, validation, and test samples unexpectedly overlap.')
    if set.union(*groups) != set(df['image']):
        raise RuntimeError('The splits do not cover all samples.')
    for subset in (train,val,test):
        subset['was_used_for_training'] = subset['image'].isin(groups[0])
    return train, val, test

def save_manifest(train, val, test, path):
    records = pd.concat([x.assign(subset=name) for x,name in
                        ((train,'train'),(val,'validation'),(test,'test'))])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(path,index=False,encoding='utf-8-sig')
