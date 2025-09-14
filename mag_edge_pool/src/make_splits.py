import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

#  ╭──────────────────────────────────────────────────────────╮
#  │ Cross Validation Splits                                  │
#  ╰──────────────────────────────────────────────────────────╯
        
def make_splits(all_idxs, targets, outer_k=10, inner_k=None, holdout_test_size=0.1, seed=42, stratify=True):
    if stratify:
        outer_kfold =  StratifiedKFold(n_splits=outer_k, shuffle=True)
    else:
        outer_kfold = KFold(n_splits=outer_k, shuffle=True)
    indices = []

    # Set the random seed for reproducibility
    np.random.seed(seed)

    for train_ok_split, test_ok_split in outer_kfold.split(X=all_idxs, y=targets):
        split = {"test": all_idxs[test_ok_split], 'model_selection': []}

        train_ok_targets = targets[train_ok_split]

        if inner_k is None:  
            assert holdout_test_size is not None
            train_i_split, val_i_split = train_test_split(train_ok_split,
                                                            stratify=train_ok_targets,
                                                            test_size=holdout_test_size)
            split['model_selection'].append(
                {"train": train_i_split, "validation": val_i_split})

        else:  
            if stratify:
                inner_kfold = StratifiedKFold(
                    n_splits=inner_k, shuffle=True)
            else:
                inner_kfold = KFold(n_splits=inner_k, shuffle=True)
            for train_ik_split, val_ik_split in inner_kfold.split(train_ok_split, train_ok_targets):
                split['model_selection'].append(
                    {"train": train_ok_split[train_ik_split], "validation": train_ok_split[val_ik_split]})
                
        idx_tr, idx_va, idx_te = split['model_selection'][0]['train'], split['model_selection'][0]['validation'], split['test']
        indices.append([idx_tr, idx_va, idx_te])
    return indices


def make_inner_split(all_idxs, targets, inner_k=10, holdout_test_size=0.1, seed=42):
    indices = []

    # Set the random seed for reproducibility
    np.random.seed(seed)

    train_i_split, val_i_split = train_test_split(all_idxs,
                                                        stratify=targets,
                                                        test_size=holdout_test_size)
    indices.append(train_i_split, val_i_split)

    return indices