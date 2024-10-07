import pandas as pd


def get_label(x):
    label = []
    groups = x.list_symmetry_groups[1:-1].split()
    train_aug = x.train_aug[1:-1].split()
    test_aug = x.test_aug[1:-1].split()
    if all(g =='id'for g in groups):
        label.append('CNN')
    elif all(g =='rot90'for g in groups):
        label.append('Multi-GCNN')
    else:
        raise NotImplementedError

    if not all(g =='id'for g in train_aug):
        label.append('Aug')

    if x.fusion:
        label.append('Fusion')

    if not all(g == 'id' for g in test_aug):
        label.append('TestAug')
    return '-'.join(label)


def get_index(x):
    index = []
    groups = x.list_symmetry_groups[1:-1].split()
    train_aug = x.train_aug[1:-1].split()
    test_aug = x.test_aug[1:-1].split()
    if all(g =='id'for g in groups):
        index.append(0)
    elif all(g =='rot90'for g in groups):
        index.append(1)
    else:
        raise NotImplementedError

    if all(g =='id'for g in train_aug):
        index.append(0)
    else:
        index.append(1)

    if x.fusion:
        index.append(1)
    else:
        index.append(0)

    if all(g == 'id' for g in test_aug):
        index.append(0)
    else:
        index.append(1)

    return index


df = pd.read_csv('runs/all_results.csv')
averaged = []
for name, group in df.groupby(['dataset', 'num_inputs', 'train_aug', 'test_aug', 'model_name', 'list_symmetry_groups', 'fusion', 'input_size']):
    group['seed'] = len(group)
    group.best_acc = group.best_acc.mean()
    group.drop_duplicates(inplace=True)
    averaged.append(group)
averaged = pd.concat(averaged)

i = 0
for (dataset, num_inputs, input_size), group in averaged.groupby(['dataset', 'num_inputs', 'input_size']):
    for train_aug, subgroup in group.groupby('train_aug'):
        subgroup = subgroup.filter(items=['seed', 'best_acc', 'list_symmetry_groups', 'train_aug', 'test_aug', 'dataset', 'fusion', 'num_inputs', ])
        subgroup['label'] = subgroup.apply(lambda x: get_label(x), axis=1)
        subgroup.index = subgroup.apply(lambda x: get_index(x), axis=1)
        subgroup.sort_index(inplace=True)
        G = subgroup.num_inputs.unique().item()

        assert subgroup.seed.unique().item() == 3
        if i == 0:
            print(','.join(['dataset', 'G', 'train_aug', ] + subgroup['label'].tolist()))
        print(','.join([dataset, str(G), str(train_aug), ] + subgroup['best_acc'].round(3).apply(str).tolist()))
        i += 1

