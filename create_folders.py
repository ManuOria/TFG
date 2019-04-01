# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:05:22 2019

@author: manfe
"""
import os
import shutil
from sklearn.model_selection import train_test_split
import glob
import xml.etree.ElementTree as ET

GLOBAL_IMAGES_PATH = os.getcwd() + '/Global/'    
GLOBAL_IMAGES_PATH_COPY = os.getcwd() + '/Global_Copy/' 
TRAIN_PATH = os.getcwd() + '/training_images/' ##Cambiar a training
TEST_PATH = os.getcwd() + '/test_images/'

def main(): 
    try:
        if os.path.exists(TRAIN_PATH):
            shutil.rmtree(TRAIN_PATH)
            
        if not os.path.exists(TRAIN_PATH):
            os.mkdir(TRAIN_PATH)
            print("The training directory has been created")
            
        if os.path.exists(TEST_PATH):
            shutil.rmtree(TEST_PATH)
            
        if not os.path.exists(TEST_PATH):
            os.mkdir(TEST_PATH)
            print("The testing directory has been created")
            
        if os.path.exists(GLOBAL_IMAGES_PATH_COPY):
            shutil.rmtree(GLOBAL_IMAGES_PATH_COPY)
            
        if not os.path.exists(GLOBAL_IMAGES_PATH_COPY):
            os.mkdir(GLOBAL_IMAGES_PATH_COPY)
            print("A copy of the global directory has been created")

    except Exception as e:
        print("unable to create directory " + TRAIN_PATH + "error: " + str(e))
    
    for jpgfiles in os.listdir(GLOBAL_IMAGES_PATH):
        shutil.copy(GLOBAL_IMAGES_PATH + jpgfiles, GLOBAL_IMAGES_PATH_COPY)
    
    filename, labels = xml_features(GLOBAL_IMAGES_PATH)

    X_train, X_test, y_train, y_test = train_test_split(filename, labels, test_size=0.10)
    
    jpg = []
    for jpgfiles in os.listdir(GLOBAL_IMAGES_PATH_COPY):
        if jpgfiles.endswith('.jpg'):
            jpg.append(jpgfiles)

    i=0
    j=0
    for i in range(len(jpg)):
        for j in range(len(X_train)):
            if jpg[i] == X_train[j]:
                try:
                    shutil.move(GLOBAL_IMAGES_PATH_COPY + jpg[i], TRAIN_PATH)
                    shutil.move(GLOBAL_IMAGES_PATH_COPY + os.path.splitext(jpg[i])[0] + ".xml", TRAIN_PATH)
                except OSError:
                    pass
                
    print('Training images have been copied into the training images folder with their associated xml')
        
    jpg = []
    for jpgfiles in os.listdir(GLOBAL_IMAGES_PATH_COPY):
        if jpgfiles.endswith('.jpg'):
            jpg.append(jpgfiles)
            
    i=0
    j=0
    for i in range(len(jpg)):
        for j in range(len(X_test)):
            if jpg[i] == X_test[j]:
                try:
                    shutil.move(GLOBAL_IMAGES_PATH_COPY + jpg[i], TEST_PATH)
                    shutil.move(GLOBAL_IMAGES_PATH_COPY + os.path.splitext(jpg[i])[0] + ".xml", TEST_PATH)
                except OSError:
                    pass
    print('Test images have been copied into the test images folder')


def xml_features(path):
    names = []
    labeling = []
    for xml_file in glob.glob(path + '/*.xml'):      
        # represent the xml as a tree, and import the data
        tree = ET.parse(xml_file)  
        root = tree.getroot()
        # extracts first the filename and the size of the image, and then the values from the 'object' of the xml
        for member in root.findall('object'):
            name = root.find('filename').text
            label = member[0].text
            names.append(name)
            labeling.append(label)
    return names, labeling


if __name__ == "__main__":
    main()
