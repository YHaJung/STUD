import os
import json, pickle

def make_path(filepath):
    file_dirs = filepath.split('/')[:-1]

    file_dir = '/' if filepath[0] == '/'else ''
    for dir in file_dirs:
        file_dir = os.path.join(file_dir, dir)

    # print(file_dir)
    os.makedirs(file_dir, exist_ok=True)

def save_file(data, filepath):
    make_path(filepath)

    extension = filepath.split('.')[-1]
    if extension == 'json':
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent="\t")
    elif extension == 'pickle' or extension=='pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=4)

def load_file(filepath):
    extension = filepath.split('.')[-1]
    extensions = {'json':json, 'pickle':pickle, 'pkl':pickle}
    with open(filepath, 'rb') as file:
        results = extensions[extension].load(file)
    
    return results


if __name__=='__main__':
    path = os.path.join('ood/threshold/', f'cluster_backbone_arch_args.train_data_args.test_data.json')
    
    make_path(path)