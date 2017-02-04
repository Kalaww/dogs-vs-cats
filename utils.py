import os
import numpy as np
import bcolz


def move_validset_into_trainset(path, verbose=False):
    """
    Get all file from validation set into training set
    :param path: parent directory containing train and valid directory
    :param verbose: print number of files moved
    :return:
    """
    train_path = os.path.join(path, 'train')
    valid_path = os.path.join(path, 'valid')
    for sub_dir in os.listdir(valid_path):
        imgs = os.listdir(os.path.join(valid_path, sub_dir))
        for img_file in imgs:
            os.rename(
                os.path.join(valid_path, sub_dir, img_file),
                os.path.join(train_path, sub_dir, img_file)
            )
        if verbose:
            print('{}\t{}'.format(len(imgs), sub_dir))


def generate_validation_set(path, val_split=0.1, verbose=False):
    """
    Generate validation set by splitting training set classes
    at random
    :param path: parent directory containing train and valid directory
    :param val_split: split value for validation set (0.0 - 1.0)
    :param verbose: print number of files moved
    :return:
    """
    train_dir = os.path.join(path, 'train')
    valid_dir = os.path.join(path, 'valid')
    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)

    for sub_dir in os.listdir(train_dir):
        p = os.path.join(valid_dir, sub_dir)
        if not os.path.exists(p):
            os.mkdir(p)
        imgs = os.listdir(os.path.join(train_dir, sub_dir))
        val_size = int(len(imgs) * val_split)
        for img in np.random.permutation(imgs)[:val_size]:
            os.rename(
                os.path.join(train_dir, sub_dir, img),
                os.path.join(p, img)
            )
        if verbose:
            print('{}/{}\t\t{}'.format(val_size, len(imgs), sub_dir))

def save_array(filename, array):
    c = bcolz.carray(array, rootdir=filename, mode='w')
    c.flush()

def load_array(filename):
    return bcolz.open(filename)[:]


