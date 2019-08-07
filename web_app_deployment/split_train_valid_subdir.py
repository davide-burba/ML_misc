
import shutil
import numpy as np
import os

def split_train_valid_subdir(data_dir = "data", valid_fraction = 0.2, valid_dir = "valid",train_dir = 'train'):
    
    os.mkdir(valid_dir)
    os.mkdir(train_dir)

    labels = [v for v in os.listdir(data_dir) if '.' not in v ]
    for subdirectory in labels:

        os.mkdir(valid_dir + '/' + subdirectory)
        os.mkdir(train_dir + '/' + subdirectory)

        subdir_content = os.listdir(data_dir + '/' + subdirectory)

        for file in subdir_content:
            if np.random.rand(1) < valid_fraction:
                shutil.copy(data_dir + '/'+ subdirectory + '/' + file, valid_dir + '/'+ subdirectory + '/' + file)
            else:
                shutil.copy(data_dir + '/'+ subdirectory + '/' + file, train_dir + '/'+ subdirectory + '/' + file)
    return


def split_train_valid_test_subdir(data_dir = "data", valid_fraction = 0.2, test_fraction = 0.2, valid_dir = "valid", train_dir = 'train', test_dir = 'test'):
    
    os.mkdir(valid_dir)
    os.mkdir(train_dir)
    os.mkdir(test_dir)

    labels = [v for v in os.listdir(data_dir) if '.' not in v ]
    for subdirectory in labels:

        os.mkdir(valid_dir + '/' + subdirectory)
        os.mkdir(train_dir + '/' + subdirectory)
        os.mkdir(test_dir + '/' + subdirectory)

        subdir_content = os.listdir(data_dir + '/' + subdirectory)

        for file in subdir_content:
            r = np.random.rand(1)
            if  r < valid_fraction:
                shutil.copy(data_dir + '/'+ subdirectory + '/' + file, valid_dir + '/'+ subdirectory + '/' + file)
            elif r < valid_fraction + test_fraction:
                shutil.copy(data_dir + '/'+ subdirectory + '/' + file, test_dir + '/'+ subdirectory + '/' + file)
            else:
                shutil.copy(data_dir + '/'+ subdirectory + '/' + file, train_dir + '/'+ subdirectory + '/' + file)
    return

