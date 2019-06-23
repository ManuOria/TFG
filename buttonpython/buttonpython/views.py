from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.core.files.uploadedfile import InMemoryUploadedFile
import numpy as np
import os
import tensorflow as tf
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util
from distutils.version import StrictVersion
from PIL import Image
from . import model
import sys
from io import BytesIO

def upload(request):
    context = {}
    if request.method == 'POST':
        uploaded_file = request.FILES['document']
        image_np = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        print(type(uploaded_file))
        #image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        fs = FileSystemStorage()
        #test_image = new_image(uploaded_file)
        test_image = make_test(image_np)
        im_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite("new.jpg", test_image)
        cv2.waitKey(25)
        #print(test_image)
        test_image_encode = Image.fromarray(np.squeeze(im_rgb))
        print(test_image_encode)
        output = BytesIO()
        test_image_encode.save(output, format = "JPEG", quality = 100)
        output.seek(0)
        new_name = uploaded_file.name.split('.')[0] + "_result.jpg"
        print(new_name)
        out_img = InMemoryUploadedFile(output,'ImageField',new_name , 'image/jpeg', sys.getsizeof(output), None)
        #test_image_encode = Image.fromarray(test_image)
        #print(test_image)
        #print(test_image_encode)
        #name = fs.save(uploaded_file.name, test_image_encode)
        name = fs.save(new_name, out_img)
        context['url'] = fs.url(name) 
    return render(request, 'upload.html', context)


def new_image(image):
    image_np = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    return image_np


def make_test(image_np): 
    # Label maps map indices to category names.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
    categories = label_map_util.convert_label_map_to_categories(model.label_map, max_num_classes=12,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with model.detection_graph.as_default():
        with tf.Session(graph=model.detection_graph) as sess:

            # Definite input and output Tensors for detection_graph
            image_tensor = model.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = model.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = model.detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = model.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = model.detection_graph.get_tensor_by_name('num_detections:0')

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            #print(image_np_expanded)
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
    
    return image_np
