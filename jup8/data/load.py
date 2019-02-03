import os
import logging
import pandas as pd
import numpy as np
from pymongo import MongoClient
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

DEFAULT_DB_NAME = 'mstar2'
DEFAULT_COLLECTION_NAME = 'targets'
#IMAGE_DIRECTORY = "/projects/jupiter8/data/train"
IMAGE_DIRECTORY = "/home/jason/data/mstar/all_targets/"
TRAIN_DEP_ANGLE = '17_DEG'
TEST_DEP_ANGLE = '15_DEG'
VALID_EXTENSIONS = ['.jpg']

client = MongoClient()
db = client[DEFAULT_DB_NAME]
collection = db[DEFAULT_COLLECTION_NAME]


def get_samples_df(query, qty=None, reindex=False):
    """Returns a data frame of samples

    Args:
        query:   query dictionary for DB
        qty:     number of samples PER CLASS to return
        reindex: reindexes dataframe from 0 to n 
    """
    cursor = collection.find(query)
    if cursor.count() < 1:
        raise RuntimeError("no results")
    # Only include files with a valid extension
    res = filter(lambda x: os.path.splitext(x['filename'])[1] in VALID_EXTENSIONS,
        list(cursor))
    df = pd.DataFrame(res)
    # Strip path from filenames
    df['filename'] = df.filename.apply(os.path.basename)
    if qty is None: # return all samples
        samples = df
    else: # return a random samples of the results
        classes = list(df.target_class.unique())
        samples = [df[df['target_class']==c].sample(qty, replace=False) for c in classes]
        samples = pd.concat(samples)
    if reindex: # reindex dataframe starting at 0
        samples.reset_index(drop=True, inplace=True)
    return samples


def get_training_generator(img_shape, batch_size=32, qty=None, exclude_synths=True):
    """Returns a training and validation data generators"""
    query = {'depression_angle': TRAIN_DEP_ANGLE}
    if exclude_synths:
        query['target_instance'] = {'$not': {'$in':['synth']}}
    df = get_samples_df(query, qty=qty)
    df_train, df_val = train_test_split(df)

    gen_train = image.ImageDataGenerator(rescale=1./255,
                                         rotation_range=20,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         horizontal_flip=True,
                                         data_format='channels_last')

    train = gen_train.flow_from_dataframe(df_train,
                                          IMAGE_DIRECTORY,
                                          x_col='filename',
                                          y_col='target_class',
                                          color_mode='grayscale',
                                          target_size=img_shape,
                                          class_mode='categorical',
                                          batch_size=batch_size)

    gen_val = image.ImageDataGenerator(rescale=1./255,
                                       data_format='channels_last')

    validate = gen_val.flow_from_dataframe(df_val,
                                           IMAGE_DIRECTORY,
                                           x_col='filename',
                                           y_col='target_class',
                                           color_mode='grayscale',
                                           target_size=img_shape,
                                           class_mode='categorical',
                                           batch_size=batch_size)
    
    return (train, validate)

def get_test_generator(img_shape, batch_size=32, qty=None, exclude_synths=True):
    """Returns a validation data generator"""
    query = {'depression_angle': TEST_DEP_ANGLE}
    if exclude_synths:
        query['target_instance'] = {'$not': {'$in':['synth']}}
    df = get_samples_df(query, qty=qty)

    gen = image.ImageDataGenerator(rescale=1./255,
                                   data_format='channels_last')

    test = gen.flow_from_dataframe(df,
                                   x_col='filename',
                                   y_col='target_class',
                                   color_mode='grayscale',
                                   target_size=img_shape,
                                   class_mode='categorical',
                                   batch_size=batch_size)
    return test
