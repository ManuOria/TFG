# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:57:14 2019

@author: manfe
"""
import os

MODEL = 'rfcn_resnet101_coco' ## 'faster_rcnn_resnet101_coco'
PIPELINE_CONFIG_PATH = os.getcwd() + "/" + MODEL + ".config" 
MODEL_DIR = os.getcwd() + "/" + MODEL + "_2018_01_28" 

###################################################### 1.

# train and test directories
TRAINING_IMAGES_DIR = os.getcwd() + "/training_images/"

TEST_IMAGES_DIR = os.getcwd() + "/test_images/"

# output .csv file names/locations
TRAINING_DATA_DIR = os.getcwd() + "/" + "training_data"
TRAIN_CSV_FILE_LOC = TRAINING_DATA_DIR + "/" + "train_labels.csv"

######################################################## 2.

# training and testing output .tfrecord files
TRAIN_TFRECORD_FILE_LOC = os.getcwd() + "/training_data/" + "train.tfrecord"

########################################################## 3.
# verify that your MODEL_DIR contains these files
FILES_MODEL_DIR_MUST_CONTAIN = [ "checkpoint" ,
                                 "frozen_inference_graph.pb",
                                 "model.ckpt.data-00000-of-00001",
                                 "model.ckpt.index",
                                 "model.ckpt.meta"]

########################################################### 4.

# the output directory to place the inference graph data, note that it's ok if this directory does not already exist
# because the call to export_inference_graph() below will create this directory if it does not exist already
OUTPUT_DIR = os.getcwd() + "/" + "inference_graph"

########################################################### 5.
TEST_IMAGE_DIR = os.getcwd() +  "/test_images"
FROZEN_INFERENCE_GRAPH_LOC = os.getcwd() + "/inference_graph/frozen_inference_graph.pb"
LABELS_LOC = os.getcwd() + "/" + "label_map.pbtxt"
RESULTS_LOC = os.getcwd() + "/result_images"
