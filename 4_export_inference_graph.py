# 4_export_inference_graph.py
#
# original file by Google:
# https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py

import os
import re
import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2
import config

# module-level variables ##############################################################################################
# the final checkpoint result of the training process
items = os.listdir(config.TRAINING_DATA_DIR)

fileList = []
for names in items:
    if names.endswith(".index"):
        fileList.append(names)

number = []  
i = 0
for i in range(len(fileList)):
    m = re.match('model.ckpt-([0-9]+).index', fileList[i])
    select = m.group(1)
    number.append(int(select))

model = max(number)

TRAINED_CHECKPOINT_PREFIX_LOC = os.getcwd() + "/training_data/model.ckpt-" + str(model)

# INPUT_TYPE can be "image_tensor", "encoded_image_string_tensor", or "tf_example"
INPUT_TYPE = "image_tensor"

# If INPUT_TYPE is "image_tensor", INPUT_SHAPE can explicitly set.  The shape of this input tensor to a fixed size.
# The dimensions are to be provided as a comma-separated list of integers. A value of -1 can be used for unknown dimensions.
# If not specified, for an image_tensor, the default shape will be partially specified as [None, None, None, 3]
INPUT_SHAPE = None

#######################################################################################################################
def main(_):
    print("starting script . . .")

    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if

    print("calling TrainEvalPipelineConfig() . . .")
    trainEvalPipelineConfig = pipeline_pb2.TrainEvalPipelineConfig()

    print("checking and merging " + os.path.basename(config.PIPELINE_CONFIG_PATH) + " into trainEvalPipelineConfig . . .")
    with tf.gfile.GFile(config.PIPELINE_CONFIG_PATH, 'r') as f:
        text_format.Merge(f.read(), trainEvalPipelineConfig)
    # end with

    print("calculating input shape . . .")
    if INPUT_SHAPE:
        input_shape = [ int(dim) if dim != '-1' else None for dim in INPUT_SHAPE.split(',') ]
    else:
        input_shape = None
    # end if

    print("calling export_inference_graph() . . .")
    exporter.export_inference_graph(INPUT_TYPE, trainEvalPipelineConfig, TRAINED_CHECKPOINT_PREFIX_LOC, config.OUTPUT_DIR, input_shape)

    print("done !!")
# end main

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(config.PIPELINE_CONFIG_PATH):
        print('ERROR: PIPELINE_CONFIG_PATH "' + config.PIPELINE_CONFIG_PATH + '" does not seem to exist')
        return False
    # end if

    # TRAINED_CHECKPOINT_PREFIX_LOC is a special case because there is no actual file with this name.
    # i.e. if TRAINED_CHECKPOINT_PREFIX_LOC is:
    # "~\training_data\model.ckpt-500"
    # this exact file does not exist, but there should be 3 files including this name, which would be:
    # "model.ckpt-500.data-00000-of-00001"
    # "model.ckpt-500.index"
    # "model.ckpt-500.meta"
    # therefore it's necessary to verify that the stated directory exists and then check if there are at least three files
    # in the stated directory that START with the stated name

    # break out the directory location and the file prefix
    trainedCkptPrefixPath, filePrefix = os.path.split(TRAINED_CHECKPOINT_PREFIX_LOC)

    # return false if the directory does not exist
    if not os.path.exists(trainedCkptPrefixPath):
        print('ERROR: directory "' + trainedCkptPrefixPath + '" does not seem to exist')
        print('was the training completed successfully?')
        return False
    # end if

    # count how many files in the stated directory start with the stated prefix
    numFilesThatStartWithPrefix = 0
    for fileName in os.listdir(trainedCkptPrefixPath):
        if fileName.startswith(filePrefix):
            numFilesThatStartWithPrefix += 1
        # end if
    # end if

    # if less than 3 files start with the stated prefix, return false
    if numFilesThatStartWithPrefix < 3:
        print('ERROR: 3 files statring with "' + filePrefix + '" do not seem to be present in the directory "' + trainedCkptPrefixPath + '"')
        print('was the training completed successfully?')
    # end if

    # if we get here the necessary directories and files are present, so return True
    return True
# end function

#######################################################################################################################
if __name__ == '__main__':
    tf.app.run()
