# 5_test.py
#
# original source from Google:
# https://github.com/tensorflow/models/blob/master/research/object_detection/test.py

import numpy as np
import os
import tensorflow as tf
import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util
from distutils.version import StrictVersion
import config

# module level variables ##############################################################################################
NUM_CLASSES = 10

#######################################################################################################################
def main():
    print("starting program . . .")
    n = 0
    
    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if

    # this next comment line is necessary to avoid a false PyCharm warning
    # noinspection PyUnresolvedReferences
    if StrictVersion(tf.__version__) < StrictVersion('1.5.0'):
        raise ImportError('Please upgrade your tensorflow installation to v1.5.* or later!')
    # end if

    # load a (frozen) TensorFlow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(config.FROZEN_INFERENCE_GRAPH_LOC, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        # end with
    # end with

    # Loading label map
    # Label maps map indices to category names.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(config.LABELS_LOC)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    imageFilePaths = []
    imagesName = []
    for imageFileName in os.listdir(config.TEST_IMAGE_DIR):
        if imageFileName.endswith(".jpg"):
            imageFilePaths.append(config.TEST_IMAGE_DIR + "/" + imageFileName)
            imagesName.append(imageFileName)
        # end if
    # end for

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in imageFilePaths:

                print(image_path)

                image_np = cv2.imread(image_path)

                if image_np is None:
                    print("error reading file " + image_path)
                    continue
                # end if

                # Definite input and output Tensors for detection_graph
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                                   np.squeeze(boxes),
                                                                   np.squeeze(classes).astype(np.int32),
                                                                   np.squeeze(scores),
                                                                   category_index,
                                                                   use_normalized_coordinates=True,
                                                                   line_thickness=8)
                
                cv2.imwrite(os.path.join(config.RESULTS_LOC ,imagesName[n]), image_np)
                n = n + 1
            # end for
        # end with
    # end with
# end main

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(config.TEST_IMAGE_DIR):
        print('ERROR: TEST_IMAGE_DIR "' + config.TEST_IMAGE_DIR + '" does not seem to exist')
        return False
    # end if
    if not os.path.exists(config.RESULTS_LOC):
            os.makedirs(config.RESULTS_LOC)
            print("Directory Created")
    # ToDo: check here that the test image directory contains at least one image

    if not os.path.exists(config.FROZEN_INFERENCE_GRAPH_LOC):
        print('ERROR: FROZEN_INFERENCE_GRAPH_LOC "' + config.FROZEN_INFERENCE_GRAPH_LOC + '" does not seem to exist')
        print('was the inference graph exported successfully?')
        return False
    # end if

    if not os.path.exists(config.LABELS_LOC):
        print('ERROR: the label map file "' + config.LABELS_LOC + '" does not seem to exist')
        return False
    # end if

    return True
# end function

#######################################################################################################################
if __name__ == "__main__":
    main()
