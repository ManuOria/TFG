import tensorflow as tf
import os
from utils import label_map_util

# load a (frozen) TensorFlow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(os.getcwd() + '/buttonpython/model_configuration/inference_graph/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        print('model imported')
    # end with
# end with
# Loading label map
label_map = label_map_util.load_labelmap(os.getcwd() + "/buttonpython/model_configuration/label_map.pbtxt")
print('labels imported')
