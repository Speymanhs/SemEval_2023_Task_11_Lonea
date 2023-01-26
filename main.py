# Import the Libraries
import os
import shutil

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa
from official.nlp import optimization  # to create AdamW optimizer
import matplotlib.pyplot as plt
import random
# importing the zipfile module
from zipfile import ZipFile
from reading_dataset_stormfront import get_500_stormfront_non_hateful_data
from reading_dataset_stormfront import get_stormfront_hateful_data

tf.get_logger().setLevel('ERROR')


# Preparing the Datasets
#_________________________________________________________________________________
### Unzip the Datasets
# loading the temp.zip and creating a zip object
with ZipFile("./resources/data_practicephase_cleardev.zip", 'r') as zip_oject:
    # Extracting all the members of the zip
    # into a specific location.
    zip_oject.extractall(path="./resources/")

path_to_extracted_folder = './resources/data_practicephase_cleardev/'

### Reading the HS-Brexit Dataset
#### Defining the Function for reading and preparing the datasets
#_________________________________________________________________________________

#_________________________________________________________________________________
#===== snippet 2: how to read data and save text, soft evaluation and hard evaluation in a different file for each dataset/split
# with these few lines you can loop across all datasets and splits (here only the train) and
# extract (and print) the info you need
# here we print: dataset,split,id,lang,hard_label,soft_label_0,soft_label_1,text in a tab separated format
# note: each item_id in the dataset for each split is numbered starting from "1"

import json
import pandas as pd
#print("Dataset\tSplit\tId\tLang\tHard_label\tSoft_label_0\tSoft_label_1\tText")                   # print header
def prepare_dataset(dataset_list, splits): #dataset_list the list of datasets you want the data for in a list of string(s) / splits can be ['train'], ['dev'], or ['train', 'dev']
  train_ds = []
  dev_ds = []
  for current_dataset in dataset_list:                         # loop on datasets
    for current_split in splits:                                                                 # loop on splits, here only train
      current_file = path_to_extracted_folder+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json'     # current file
      if current_split == 'train':
        train_ds.append(json.load(open(current_file,'r', encoding = 'UTF-8')))
      else:
        dev_ds.append(json.load(open(current_file,'r', encoding = 'UTF-8')))
  return train_ds, dev_ds

train_ds, dev_ds = prepare_dataset(['ArMIS'], ['train', 'dev'])
print(dev_ds)

#### Convert the datasets from dictionaries to dataframe
train_df = pd.DataFrame.from_dict(train_ds[0], orient="index")
dev_df = pd.DataFrame.from_dict(dev_ds[0], orient="index")

## Creating a list from hard labels
# Convert the labels from dictionary to tensor
train_hard_labels = [float(soft_label) for soft_label in train_df['hard_label']]
#train_soft_labels = tf.convert_to_tensor(train_soft_labels)

dev_hard_labels = [float(soft_label) for soft_label in dev_df['hard_label']]
#dev_soft_labels = tf.convert_to_tensor(dev_soft_labels)
print(train_hard_labels)
# print(sum(train_hard_labels)/len(train_hard_labels))
# a = {0:1, 1:0}
# for i in range(len(train_hard_labels)):
#   train_hard_labels[i] = a[train_hard_labels[i]]
# for i in range(len(dev_hard_labels)):
#   dev_hard_labels[i] = a[dev_hard_labels[i]]
# print(train_hard_labels)
# print(sum(train_hard_labels)/len(train_hard_labels))


# Convert the text dataframes to list (both training and dev)
tf_train_text = train_df['text']
tf_train_list = [text for text in tf_train_text]
tf_dev_text = dev_df['text']
tf_dev_list = [text for text in tf_dev_text]
#_________________________________________________________________________________


#_________________________________________________________________________________
##Reading Additional Hateful Data from Stormfront Dataset and appending it to the training set
list_of_additional_data, hard_labels_of_additional_data = get_stormfront_hateful_data()
list_of_additional_data_2, hard_labels_of_additional_data_2 = get_500_stormfront_non_hateful_data()
tf_train_list = tf_train_list + list_of_additional_data + list_of_additional_data_2
train_hard_labels = train_hard_labels + hard_labels_of_additional_data + hard_labels_of_additional_data_2
#_________________________________________________________________________________


#_________________________________________________________________________________
### Re-shuffling the new list so the syntehsized data is distributed across all parts of dataset
compact_list = list(zip(tf_train_list, train_hard_labels))

random.shuffle(compact_list)

tf_train_list, train_hard_labels = zip(*compact_list)

print(sum(train_hard_labels)/len(train_hard_labels))
#_________________________________________________________________________________




# Model

#_________________________________________________________________________________
##Defining BERT pre-processor and encoder

# tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
# tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'

#For bert with talking head
# tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
# tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1'

#For Bert Base:
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
#_________________________________________________________________________________


#_________________________________________________________________________________
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
  return tf.keras.Model(text_input, net)
#_________________________________________________________________________________


#_________________________________________________________________________________
##Defining the Metric and Loss Function
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

## Defining Parameters and Optimizer
epochs = 50
steps_per_epoch = len(tf_train_list)
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
#_________________________________________________________________________________


#_________________________________________________________________________________
## Building and Compiling the Model
# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# Open a strategy scope.
with strategy.scope():
  # Everything that creates variables should be under the strategy scope.
  # In general this is only model construction & `compile()`.
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')
    classifier_model = build_classifier_model()
    classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics= tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='micro'))

#_________________________________________________________________________________


#_________________________________________________________________________________
from sklearn.utils import class_weight, compute_class_weight
import numpy as np
class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(train_hard_labels),
                                        y = train_hard_labels
                                    )
class_weights = dict(zip(np.unique(train_hard_labels), class_weights))
class_weights
#_________________________________________________________________________________


#_________________________________________________________________________________
## Training the Model
### Model checkpoint
checkpoint = tf.keras.callbacks.ModelCheckpoint('./models/ConvAbuse_best_f1_weights.h5', monitor="f1_score", mode="max", save_best_only=True, save_weights_only=1, verbose=1)
classifier_model.fit(x=tf_train_list, y=train_hard_labels, validation_data=(tf_dev_list, dev_hard_labels), epochs=epochs, class_weight=class_weights, callbacks=[checkpoint])
#_________________________________________________________________________________


#_________________________________________________________________________________
#### Continue training a good model
fin_list, fin_labels = get_dynamically_generated_data()
#Continue training
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# Open a strategy scope.
with strategy.scope():
    trained_model = build_classifier_model()
    trained_model.load_weights('./models/best_retrained_2_f1_weights.h5')
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    trained_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='micro'))

checkpoint = tf.keras.callbacks.ModelCheckpoint('./models/best_retrained_3_f1_weights.h5', monitor="val_f1_score", mode="max", save_best_only=True, save_weights_only=1, verbose=1)

trained_model.fit(x=tf_train_list, y=train_hard_labels, validation_data=(tf_dev_list, dev_hard_labels),
                         epochs=epochs, class_weight=class_weights, callbacks=[checkpoint])
#_________________________________________________________________________________


#_________________________________________________________________________________
#Checking the performance on the dev set manually and 1 by 1 to check label sanity
trained_model = build_classifier_model()
trained_model.load_weights('./models/best_retrained_3_f1_weights.h5')
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
trained_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='micro'))

print(tf_dev_list[0])
x = 0
for i in range(len(tf_dev_list)):
    if round(tf.sigmoid(trained_model(tf.constant([tf_dev_list[i]]))).numpy().astype(float)[0][0]) != dev_hard_labels[i]:
        print("Index: ", i)
        print("The input text: ", tf_dev_list[i])
        print("The true label: ", dev_hard_labels[i])
        print("The model output: ", round(tf.sigmoid(trained_model(tf.constant([tf_dev_list[i]]))).numpy().astype(float)[0][0]))
        print("The model output: ", tf.sigmoid(trained_model(tf.constant([tf_dev_list[i]]))).numpy().astype(float)[0][0])
        x = x + 1
print(x)

a = tf.sigmoid(trained_model(tf.constant([tf_dev_list[0]]))).numpy().astype(int)[0][0]
print(a)
print(len(dev_hard_labels))
print(sum(dev_hard_labels))
#_________________________________________________________________________________

#_________________________________________________________________________________

loss, accuracy = trained_model.evaluate(x=tf_dev_list, y=dev_hard_labels)
print(f'accuracy: {accuracy}')
print(f'loss: {loss}')
#_________________________________________________________________________________
import csv
with open("./ArMIS_results.tsv", "w") as record_file:
    writer = csv.writer(record_file, delimiter='\t', lineterminator='\n')
    for i in range(len(tf_dev_list)):
        writer.writerow(["%d" % (int(round(tf.sigmoid(trained_model(tf.constant([tf_dev_list[i]]))).numpy().astype(float)[0][0])),
                                ), 0.5, 0.5])
#_________________________________________________________________________________

print(round(tf.sigmoid(trained_model(tf.constant([tf_dev_list[0]]))).numpy().astype(float)[0][0]))


#_________________________________________________________________________________
#Getting the model trained on large dataset and training it on convabuse
#Continue training
epochs = 40
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# Open a strategy scope.
with strategy.scope():
    trained_model = build_classifier_model()
    trained_model.load_weights('./models/best_ArMIS_retrained_3_f1_weights.h5')
    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    trained_model.compile(optimizer=optimizer,
                             loss=loss,
                             metrics=tfa.metrics.F1Score(num_classes=2, threshold=0.5, average='micro'))

checkpoint = tf.keras.callbacks.ModelCheckpoint('./models/best_ArMIS_retrained_3_f1_weights.h5', monitor="f1_score", mode="max", save_best_only=True, save_weights_only=1, verbose=1)
trained_model.fit(x=tf_train_list, y=train_hard_labels, validation_data=(tf_dev_list, dev_hard_labels),
                         epochs=epochs, class_weight=class_weights, callbacks=[checkpoint])


