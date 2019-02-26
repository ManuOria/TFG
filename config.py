# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:57:14 2019

@author: manfe
"""
import os

###################################################### 1.
# train and test directories
TRAINING_IMAGES_DIR = os.getcwd() + "/training_images/"

TEST_IMAGES_DIR = os.getcwd() + "/test_images/"

# output .csv file names/locations
TRAINING_DATA_DIR = os.getcwd() + "/" + "training_data"
TRAIN_CSV_FILE_LOC = TRAINING_DATA_DIR + "/" + "train_labels.csv"
EVAL_CSV_FILE_LOC = TRAINING_DATA_DIR + "/" + "eval_labels.csv"

######################################################## 2.

TRAIN_CSV_FILE_LOC = os.getcwd() + "/training_data/" + "train_labels.csv"
TRAIN_IMAGES_DIR = os.getcwd() + "/training_images"

# input test CSV file and test images directory
EVAL_CSV_FILE_LOC = os.getcwd() + "/training_data/" + "eval_labels.csv"
TEST_IMAGES_DIR = os.getcwd() + "/test_images"

# training and testing output .tfrecord files
TRAIN_TFRECORD_FILE_LOC = os.getcwd() + "/training_data/" + "train.tfrecord"
EVAL_TFRECORD_FILE_LOC = os.getcwd() + "/training_data/" + "eval.tfrecord"

########################################################## 3.

# this is the big (pipeline).config file that contains various directory locations and many tunable parameters
PIPELINE_CONFIG_PATH = os.getcwd() + "/" + "ssd_inception_v2_coco.config"

# verify this extracted directory exists,
# also verify it's the directory referred to by the 'fine_tune_checkpoint' parameter in your (pipeline).config file
MODEL_DIR = os.getcwd() + "/" + "ssd_inception_v2_coco_2018_01_28"

# verify that your MODEL_DIR contains these files
FILES_MODEL_DIR_MUST_CONTAIN = [ "checkpoint" ,
                                 "frozen_inference_graph.pb",
                                 "model.ckpt.data-00000-of-00001",
                                 "model.ckpt.index",
                                 "model.ckpt.meta"]

# directory to save the checkpoints and training summaries
TRAINING_DATA_DIR = os.getcwd() + "/training_data/"

########################################################### 4.

# the location of the big config file
PIPELINE_CONFIG_LOC =  os.getcwd() + "/" + "ssd_inception_v2_coco.config"

# the final checkpoint result of the training process
TRAINED_CHECKPOINT_PREFIX_LOC = os.getcwd() + "/training_data/model.ckpt-20"

# the output directory to place the inference graph data, note that it's ok if this directory does not already exist
# because the call to export_inference_graph() below will create this directory if it does not exist already
OUTPUT_DIR = os.getcwd() + "/" + "inference_graph"

########################################################### 5.
TEST_IMAGE_DIR = os.getcwd() +  "/test_images"
FROZEN_INFERENCE_GRAPH_LOC = os.getcwd() + "/inference_graph/frozen_inference_graph.pb"
LABELS_LOC = os.getcwd() + "/" + "label_map.pbtxt"