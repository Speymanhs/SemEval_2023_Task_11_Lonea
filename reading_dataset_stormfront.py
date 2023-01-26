### Unzip the Dataset

# importing the zipfile module
from zipfile import ZipFile
import pandas as pd
import random

# loading the temp.zip and creating a zip object
with ZipFile("./resources/Sentences_from_Stormfront_dataset.zip", 'r') as zip_oject:
    # Extracting all the members of the zip
    # into a specific location.
    zip_oject.extractall(path="./resources/")

path_to_extracted_folder = './resources/Sentences_from_Stormfront_dataset/'
path_to_data_files = './resources/Sentences_from_Stormfront_dataset/all_files/'


def get_500_stormfront_non_hateful_data():
    annotation_metadata = pd.read_csv(path_to_extracted_folder + 'annotations_metadata.csv')
    list_of_additional_data = []
    hard_labels_of_additional_data = []
    for i in range(500):
        rand_num = random.randint(0, len(annotation_metadata) - 1)
        while annotation_metadata['label'][rand_num] == 'hate':
            rand_num = random.randint(0, len(annotation_metadata) - 1)
        file_address = path_to_data_files + annotation_metadata['file_id'][rand_num] + '.txt'
        text = open(file_address, "r").read()
        list_of_additional_data.append(text)
        hard_labels_of_additional_data.append(0.0)
    return list_of_additional_data, hard_labels_of_additional_data


def get_stormfront_hateful_data():
    annotation_metadata = pd.read_csv(path_to_extracted_folder + 'annotations_metadata.csv')
    list_of_additional_data = []
    hard_labels_of_additional_data = []

    for i in range(len(annotation_metadata)):
        if annotation_metadata['label'][i] == 'hate':
            file_address = path_to_data_files + annotation_metadata['file_id'][i] + '.txt'
            text = open(file_address, "r").read()
            list_of_additional_data.append(text)
            hard_labels_of_additional_data.append(1.0)
    return list_of_additional_data, hard_labels_of_additional_data
