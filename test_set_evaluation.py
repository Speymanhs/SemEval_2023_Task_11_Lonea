#Block 1: Importing the Libraries and required functions
#_________________________________________________________________________________
import os
import shutil
import math
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa
from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt
import random
from zipfile import ZipFile
from reading_dataset_stormfront import get_500_stormfront_non_hateful_data
from reading_dataset_stormfront import get_stormfront_hateful_data
import json
import pandas as pd
from sklearn.utils import class_weight, compute_class_weight
import numpy as np
tf.get_logger().setLevel('ERROR')
#_________________________________________________________________________________


#Block 1.5: Setting only one GPU as the visible ones
#_________________________________________________________________________________
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
#_________________________________________________________________________________





#Block 2: Defining the functions for preparing the desired dataset in a list form
#_________________________________________________________________________________
#print("Dataset\tSplit\tId\tLang\tHard_label\tSoft_label_0\tSoft_label_1\tText")
def prepare_dataset(dataset_list, splits):
  path_to_extracted_folder = './resources/data_practicephase_cleardev/'
  train_ds = []
  dev_ds = []
  test_ds = []
  for current_dataset in dataset_list:                         # loop on datasets
    for current_split in splits:                                                                 # loop on splits, here only train
      current_file = path_to_extracted_folder+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json'     # current file
      if current_split == 'train':
        train_ds.append(json.load(open(current_file,'r', encoding = 'UTF-8')))
      elif current_split == 'dev':
        dev_ds.append(json.load(open(current_file,'r', encoding = 'UTF-8')))
      else:
        test_ds.append(json.load(open(current_file, 'r', encoding='UTF-8')))
  return train_ds, dev_ds, test_ds



def convert_data_to_list(train_ds, dev_ds, test_ds):
  # Convert the datasets from dictionaries to dataframe
  train_df = pd.DataFrame.from_dict(train_ds[0], orient="index")
  dev_df = pd.DataFrame.from_dict(dev_ds[0], orient="index")
  test_df = pd.DataFrame.from_dict(test_ds[0], orient="index")

  train_hard_labels = [float(hard_label) for hard_label in train_df['hard_label']]
  train_soft_labels = [float(soft_label['1']) for soft_label in train_df['soft_label']]

  dev_hard_labels = [float(hard_label) for hard_label in dev_df['hard_label']]
  dev_soft_labels = [float(soft_label['1']) for soft_label in dev_df['soft_label']]

  test_hard_labels = [float(hard_label) for hard_label in test_df['hard_label']]
  test_soft_labels = [float(soft_label['1']) for soft_label in test_df['soft_label']]

  # Convert the text dataframes to list (both training and dev)
  tf_train_text = train_df['text']
  tf_train_list = [text for text in tf_train_text]
  tf_dev_text = dev_df['text']
  tf_dev_list = [text for text in tf_dev_text]
  tf_test_text = test_df['text']
  tf_test_list = [text for text in tf_test_text]

  return tf_train_list, train_hard_labels, train_soft_labels, \
      tf_dev_list, dev_hard_labels, dev_soft_labels, \
      tf_test_list, test_hard_labels, test_soft_labels,
#_________________________________________________________________________________





#Block 3: Calling for relevant dataset to be prepared
#_________________________________________________________________________________
train_ds, dev_ds, test_ds = prepare_dataset(['HS-Brexit'], ['train', 'dev', 'test'])
tf_train_list, train_hard_labels, \
  train_soft_labels, tf_dev_list, \
  dev_hard_labels, dev_soft_labels, \
    tf_test_list, test_hard_labels, \
    test_soft_labels, = convert_data_to_list(train_ds, dev_ds, test_ds)
#_________________________________________________________________________________




#Block 6: Choosing the preprocessor and encoder for the model from relevant repo
#_________________________________________________________________________________
#Choosing preprocessor and encoder for BERT Base
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'

#_________________________________________________________________________________





#Block 7: Defining the model architecture
#_________________________________________________________________________________
from activation_functions import step_discrete_func, step_sin_activation
##Defining the Model
def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.2)(net)
  net = tf.keras.layers.Dense(20, activation='relu', name='classifier')(net)
  net = tf.keras.layers.Dropout(0.15)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier_2')(net)
  # net = step_sin_activation(net * 0.2, 5, 0.0)
  net = tf.math.sigmoid(tf.math.divide(net, 5))
  #net = tf.math.maximum(tf.math.minimum(0.05 * net + 0.5, 1), 0)
  return tf.keras.Model(text_input, net)
#_________________________________________________________________________________





#Block 8: Defining model parameters and functions
#_________________________________________________________________________________
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
metrics = tf.metrics.BinaryAccuracy()
epochs = 35
steps_per_epoch = len(tf_train_list)
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)
init_lr = 3e-5

class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(train_soft_labels),
                                        y = train_soft_labels
                                    )
class_weights = dict(zip(np.unique(train_soft_labels), class_weights))
print(class_weights)
#_________________________________________________________________________________




#Block 10: Evaluating the model if needed
#_________________________________________________________________________________

loss = trained_model.evaluate(x=tf_test_list, y=test_soft_labels)
# print(f'accuracy: {accuracy}')
print(f'loss: {loss}')
#_________________________________________________________________________________




#Block 11: Loading the model from weights for evaluation
#_________________________________________________________________________________
trained_model = build_classifier_model()
trained_model.load_weights('./models/evaluation_phase/HS_Brexit_140_epochs_vanilla.h5')
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
trained_model.compile(optimizer=optimizer,
                      loss=loss)  # ,
#_________________________________________________________________________________





#Block 12: Preparing the tsv file for competition (if the sigmoid is present in model architecture)
#_________________________________________________________________________________
import csv
with open("./tsv_results/HS-Brexit_results.tsv", "w") as record_file:
    writer = csv.writer(record_file, delimiter='\t', lineterminator='\n')
    for i in range(len(tf_test_list)):
        writer.writerow(["%d" % (int(round(trained_model(tf.constant([tf_test_list[i]])).numpy().astype(float)[0][0]))),
                        "%f" % (1.0 - step_discrete_func(trained_model(tf.constant([tf_test_list[i]])).numpy().astype(float)[0][0], 6, 1)),
                        "%f" % step_discrete_func(trained_model(tf.constant([tf_test_list[i]])).numpy().astype(float)[0][0], 6, 1)])
#_________________________________________________________________________________





#Block 13: Preparing the tsv file for competition (if the sigmoid is present in model architecture)
#_________________________________________________________________________________
import csv
with open("./tsv_results/MD-Agreement_results.tsv", "w") as record_file:
    writer = csv.writer(record_file, delimiter='\t', lineterminator='\n')
    for i in range(len(tf_test_list)):
        writer.writerow(["%d" % (int(round(trained_model(tf.constant([tf_test_list[i]])).numpy().astype(float)[0][0]))),
                        "%f" % (1.0 - trained_model(tf.constant([tf_test_list[i]])).numpy().astype(float)[0][0]),
                        "%f" % (trained_model(tf.constant([tf_test_list[i]])).numpy().astype(float)[0][0])])
#_________________________________________________________________________________


print(len(tf_test_list))


#Block 14: Preparing the tsv file for competition (if the sigmoid is not added in model architecture)
#_________________________________________________________________________________
import csv
with open("./tsv_results/MD-Agreement_results.tsv", "w") as record_file:
    writer = csv.writer(record_file, delimiter='\t', lineterminator='\n')
    for i in range(len(tf_dev_list)):
        writer.writerow(["%d" % (int(round(tf.sigmoid(trained_model(tf.constant([tf_dev_list[i]]))).numpy().astype(float)[0][0]))),
                         "%f" % (1.0 - tf.sigmoid(trained_model(tf.constant([tf_dev_list[i]])).numpy().astype(float)[0][0])),
                         "%f" % tf.sigmoid(trained_model(tf.constant([tf_dev_list[i]])).numpy().astype(float)[0][0])])
#_________________________________________________________________________________




#Block 15: Checking which examples is our model classifying way incorrectly
#_________________________________________________________________________________
difference_vector = []
for i in range(len(tf_train_list)):
    difference_vector.append((
        abs(train_soft_labels[i] - trained_model(tf.constant([tf_train_list[i]])).numpy().astype(float)[0][0]),
        tf_train_list[i],
        train_soft_labels[i],
        trained_model(tf.constant([tf_train_list[i]])).numpy().astype(float)[0][0]))


sorted_diff = sorted(
    difference_vector,
    key=lambda x: x[0],
    reverse=True
)
print((sorted_diff[48]))
#_________________________________________________________________________________
