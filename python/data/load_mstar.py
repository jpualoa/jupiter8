# ==============================================================================
# load_mstar.py
#
# Helper functions for loading MSTAR data for test and training
#
# Author: J. Pualoa
# ==============================================================================
import os
import pymongo
import cv2
import numpy as np

MSTAR_TARGET_JPEGS_DIR = "/projects/jupiter8/images/mstar/targets"

connection = pymongo.Connection()
db = connection.mstar

def load_data(target_class, label=None, depression_angle=None, target_instance=None):
    # Build DB query
    query = {}
    query['target_class']= target_class
    if label is not None:
        query['label'] = label
    if depression_angle is not None:    
        query['depression_angle']= depression_angle
    if target_instance is not None:
        query['target_instance']= target_instance


    cursor = db.targets.find(query)
    data = []
    for c in cursor:
        img_file = os.path.join(MSTAR_TARGET_JPEGS_DIR, c['filename'])
        img = cv2.imread(img_file)
        # Image is grayscale but data is represented a BGR, just extract first channel
        data.append(img[:,:,0])
    return np.array(data)
