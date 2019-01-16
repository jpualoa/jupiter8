import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical

DEFAULT_DB_NAME = 'mstar'
DEFAULT_COLLECTION_NAME = 'targets'
CLASSES = ['T72', 'BMP2', 'BTR70']
IMAGE_DIRECTORY = "/projects/jupiter8/images/mstar/targets/"

client = MongoClient()
db = client[DEFAULT_DB_NAME]
collection = db[DEFAULT_COLLECTION_NAME]

def get_samples_df(label, qty=None, classes=CLASSES,
                   exclude_synths=True, strip_dir=True, reindex=True):
    query = {'target_class': {'$in':classes}, 'label':label}
    if exclude_synths:
        query['target_instance'] = {'$not': {'$in':['synth']}}
    cursor = collection.find(query)
    if cursor.count() < 1:
        raise RuntimeError("no results")
    df = pd.DataFrame(list(cursor))
    if strip_dir: # strip directory_path from image files
        df.filename = df.filename.apply(os.path.basename)
    if qty is None: # return all samples
        samples = df
    else: # return a random samples of the results
        samples = [df[df['target_class']==c].sample(qty) for c in classes]
        samples = pd.concat(samples)
    if reindex: # reindex dataframe starting at 0
        samples.reset_index(drop=True, inplace=True)
    return samples

def get_training_generator(batch_size=32, qty=None, exclude_synths=True):
    """Returns a training data generator"""
    gen = image.ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   data_format='channels_last')
    df = get_samples_df(label='train', qty=qty, exclude_synths=exclude_synths)
    print("\nTraining Data Samples")
    print(df.groupby('target_class').count())
    train = gen.flow_from_dataframe(df,
                                    directory=IMAGE_DIRECTORY,
                                    x_col='filename',
                                    y_col='target_class',
                                    color_mode='grayscale',
                                    target_size=(128,128),
                                    class_mode='categorical',
                                    batch_size=batch_size)
    return train

def get_test_generator(batch_size=32, qty=None, exclude_synths=True):
    """Returns a validation data generator"""
    gen = image.ImageDataGenerator(rescale=1./255,
                                   data_format='channels_last')
    df = get_samples_df(label='test', qty=qty, exclude_synths=exclude_synths)
    print("\nTest Data Samples")
    print(df.groupby('target_class').count())
    test = gen.flow_from_dataframe(df,
                                   directory=IMAGE_DIRECTORY,
                                   x_col='filename',
                                   y_col='target_class',
                                   color_mode='grayscale',
                                   target_size=(128,128),
                                   class_mode='categorical',
                                   batch_size=batch_size)
    return test
