# 1_xml_to_csv.py

# Note: substantial portions of this code, expecially the actual XML to CSV conversion, are credit to Dat Tran
# see his website here: https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
# and his GitHub here: https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import config
import shutil
import argparse

# module level variables ##############################################################################################

MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING = 10
MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING = 100

MIN_NUM_IMAGES_REQUIRED_FOR_TESTING = 5

#######################################################################################################################
def main():
    if not checkIfNecessaryPathsAndFilesExist():
        return
    # end if

    # if the training data directory does not exist, create it
    try:
        # argparse function for deleting de folders 
        parser = argparse.ArgumentParser(description='Delete Training and Inference Graph folders')
        parser.add_argument('-delete', action = "store_true", default = False, dest = 'boolean_switch', help = 'When call it deletes the folders')
        result = parser.parse_args()
        if(result.boolean_switch == True):
            if os.path.exists(config.TRAINING_DATA_DIR):
                shutil.rmtree(config.TRAINING_DATA_DIR)
                print("Training Directory Deleted")
                
            if os.path.exists(config.OUTPUT_DIR):
                shutil.rmtree(config.OUTPUT_DIR)
                print("Inference Graph Deleted")
            
        if not os.path.exists(config.TRAINING_DATA_DIR):
            os.makedirs(config.TRAINING_DATA_DIR)
            print("Training Directory Created")
        # end if
    except Exception as e:
        print("unable to create directory " + config.TRAINING_DATA_DIR + "error: " + str(e))
    # end try


    # convert training xml data to a single .csv file
    print("converting xml training data . . .")
    #Call xml_to_csv function passing the training images directory as argument, the result is stored
    trainCsvResults = xml_to_csv(config.TRAINING_IMAGES_DIR) 
    #Write a csv file at the location passed, with the information of the xml files, because the encode the information we need
    trainCsvResults.to_csv(config.TRAIN_CSV_FILE_LOC, index=None) 
    print("training xml to .csv conversion successful, saved result to " + config.TRAIN_CSV_FILE_LOC)

    # # convert test xml data to a single .csv file
    print("converting xml test data . . .")
    testCsvResults = xml_to_csv(config.TEST_IMAGES_DIR)
    testCsvResults.to_csv(config.EVAL_CSV_FILE_LOC, index=None)
    print("test xml to .csv conversion successful, saved result to " + config.EVAL_CSV_FILE_LOC)

# end main

#######################################################################################################################
def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(config.TRAINING_IMAGES_DIR):
        print('')
        print('ERROR: the training images directory "' + config.TRAINING_IMAGES_DIR + '" does not seem to exist')
        print('Did you set up the training images?')
        print('')
        return False
    # end if

    # get a list of all the .jpg / .xml file pairs in the training images directory
    # Only append those .jpg that have an associated .xml 
    trainingImagesWithAMatchingXmlFile = []
    for fileName in os.listdir(config.TRAINING_IMAGES_DIR):
        if fileName.endswith(".jpg"):
            xmlFileName = os.path.splitext(fileName)[0] + ".xml" 
            if os.path.exists(os.path.join(config.TRAINING_IMAGES_DIR, xmlFileName)):
                trainingImagesWithAMatchingXmlFile.append(fileName)
            # end if
        # end if
    # end for

    # show an error and return false if there are no images in the training directory
    if len(trainingImagesWithAMatchingXmlFile) <= 0:
        print("ERROR: there don't seem to be any images and matching XML files in " + config.TRAINING_IMAGES_DIR)
        print("Did you set up the training images?")
        return False
    # end if

    # show an error and return false if there are not at least 10 images and 10 matching XML files in TRAINING_IMAGES_DIR
    if len(trainingImagesWithAMatchingXmlFile) < MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING:
        print("ERROR: there are not at least " + str(MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING) + " images and matching XML files in " + config.TRAINING_IMAGES_DIR)
        print("Did you set up the training images?")
        return False
    # end if

    # show a warning if there are not at least 100 images and 100 matching XML files in TEST_IMAGES_DIR
    if len(trainingImagesWithAMatchingXmlFile) < MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING:
        print("WARNING: there are not at least " + str(MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING) + " images and matching XML files in " + config.TRAINING_IMAGES_DIR)
        print("At least " + str(MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING) + " image / xml pairs are recommended for bare minimum acceptable results")
        # note we do not return false here b/c this is a warning, not an error
    # end if

    return True
# end function

#######################################################################################################################
# We pass the training images path
def xml_to_csv(path):
    xml_list = []
    #glob.glob(pathname) return a list of path names that match pathname = string containing a path
    for xml_file in glob.glob(path + '/*.xml'): 
        # represent the xml as a tree, and import the data
        tree = ET.parse(xml_file)  
        root = tree.getroot()
        # extracts first the filename and the size of the image, and then the values from the 'object' of the xml
        for member in root.findall('object'): 
            value = (root.find('filename').text, int(root.find('size')[0].text), int(root.find('size')[1].text), member[0].text,
                     int(member[4][0].text), int(member[4][1].text), int(member[4][2].text), int(member[4][3].text))
            # append this information
            xml_list.append(value)
        # end for
    # end for
    # now we generate a two dimensional tabular data structure with labeled axes (the data)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
# end function

#######################################################################################################################
if __name__ == "__main__":
    main()
