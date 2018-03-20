# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:33:26 2018

@author: TapperR
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import pandas as pd

from IPython.display import Image

import tensorflow as tf
print('This code requires TensorFlow v1.3+')
print('You have:', tf.__version__)


#census_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
#census_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
census_train_path = tf.contrib.keras.utils.get_file('census.train', census_train_url)
census_test_path = tf.contrib.keras.utils.get_file('census.test', census_test_url)
#census_train_path = 'C:\\Users\\TapperR\\Desktop\\sfc\\multi-class-text-classification-cnn-rnn\\adult.csv'
#census_test_path = 'C:\\Users\\TapperR\\Desktop\\sfc\\multi-class-text-classification-cnn-rnn\\adult_test.csv'


column_names = [
  'age', 'workclass', 'fnlwgt', 'education', 'education-num',
  'marital-status', 'occupation', 'relationship', 'race', 'gender',
  'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
  'income'
]


census_train = pd.read_csv(census_train_path, index_col=False, names=column_names) 
census_test = pd.read_csv(census_test_path, skiprows=1, index_col=False, names=column_names)

census_train = census_train.dropna(how="any", axis=0)
census_test = census_test.dropna(how="any", axis=0)

census_train_label = census_train.pop('income').apply(lambda x: ">50K" in x)
census_test_label = census_test.pop('income').apply(lambda x: ">50K" in x)


print ("Training examples: %d" % census_train.shape[0])
print ("Training labels: %d" % census_train_label.shape[0])
print()
print ("Test examples: %d" % census_test.shape[0])
print ("Test labels: %d" % census_test_label.shape[0])



census_train.head()


census_train_label.head(10)


def create_train_input_fn(): 
    return tf.estimator.inputs.pandas_input_fn(
        x=census_train,
        y=census_train_label, 
        batch_size=32,
        num_epochs=None, # Repeat forever
        shuffle=True)


def create_test_input_fn():
    return tf.estimator.inputs.pandas_input_fn(
        x=census_test,
        y=census_test_label, 
        num_epochs=1, # Just one epoch
        shuffle=False) # Don't shuffle so we can compare to census_test_labels later


feature_columns = []

age = tf.feature_column.numeric_column('age')
feature_columns.append(age)




age_buckets = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('age'), 
    boundaries=[31, 46, 60, 75, 90] # specify the ranges
)

feature_columns.append(age_buckets)




education = tf.feature_column.categorical_column_with_vocabulary_list(
    "education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ])

feature_columns.append(education)




native_country = tf.feature_column.categorical_column_with_hash_bucket('native-country', 1000)
feature_columns.append(native_country)



age_cross_education = tf.feature_column.crossed_column(
    [age_buckets, education],
    hash_bucket_size=int(1e4) # Using a hash is handy here
)
feature_columns.append(age_cross_education)





train_input_fn = create_train_input_fn()
estimator = tf.estimator.LinearClassifier(feature_columns, model_dir='graphs/linear', n_classes=2)
estimator.train(train_input_fn, steps=1000)



test_input_fn = create_test_input_fn()
estimator.evaluate(test_input_fn)



predictions = estimator.predict(test_input_fn)
i = 0
for prediction in predictions:
    true_label = census_test_label[i]
    predicted_label = prediction['class_ids'][0]
    # Uncomment the following line to see probabilities for individual classes
    # print(prediction) 
    print("Example %d. Actual: %d, Predicted: %d" % (i, true_label, predicted_label))
    i += 1
    if i == 5: break


workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass',
    [' Self-emp-not-inc', ' Private', ' State-gov', ' Federal-gov',
     ' Local-gov', ' ?', ' Self-emp-inc', ' Without-pay', ' Never-worked'])

education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education',
    [' Bachelors', ' HS-grad', ' 11th', ' Masters', ' 9th', ' Some-college',
     ' Assoc-acdm', ' Assoc-voc', ' 7th-8th', ' Doctorate', ' Prof-school',
     ' 5th-6th', ' 10th', ' 1st-4th', ' Preschool', ' 12th'])

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital-status',
    [' Married-civ-spouse', ' Divorced', ' Married-spouse-absent',
     ' Never-married', ' Separated', ' Married-AF-spouse', ' Widowed'])
     
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship',
    [' Husband', ' Not-in-family', ' Wife', ' Own-child', ' Unmarried',
     ' Other-relative'])




feature_columns = [

    # Use indicator columns for low dimensional vocabularies
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(marital_status),
    tf.feature_column.indicator_column(relationship),

    # Use embedding columns for high dimensional vocabularies
    tf.feature_column.embedding_column(  # now using embedding!
        # params are hash buckets, embedding size
        tf.feature_column.categorical_column_with_hash_bucket('occupation', 100), 10),
    
    # numeric features
    tf.feature_column.numeric_column('age'),
    tf.feature_column.numeric_column('education-num'),
    tf.feature_column.numeric_column('capital-gain'),
    tf.feature_column.numeric_column('capital-loss'),
    tf.feature_column.numeric_column('hours-per-week'),   
]    
    
    


estimator = tf.estimator.DNNClassifier(hidden_units=[256, 128, 64], 
                                       feature_columns=feature_columns, 
                                       n_classes=2, 
                                       model_dir='graphs/dnn')    
    
    
    
    
train_input_fn = create_train_input_fn()
estimator.train(train_input_fn, steps=2000)   
    

test_input_fn = create_test_input_fn()
estimator.evaluate(test_input_fn)




csv_defaults = collections.OrderedDict([
  ('age',[0]),
  ('workclass',['']),
  ('fnlwgt',[0]),
  ('education',['']),
  ('education-num',[0]),
  ('marital-status',['']),
  ('occupation',['']),
  ('relationship',['']),
  ('race',['']),
  ('sex',['']),
  ('capital-gain',[0]),
  ('capital-loss',[0]),
  ('hours-per-week',[0]),
  ('native-country',['']),
  ('income',['']),
])


def csv_decoder(line):
    """Convert a CSV row to a dictonary of features."""
    parsed = tf.decode_csv(line, list(csv_defaults.values()))
    return dict(zip(csv_defaults.keys(), parsed))

# The train file has an extra empty line at the end.
# We'll use this method to filter that out.
def filter_empty_lines(line):
    return tf.not_equal(tf.size(tf.string_split([line], ',').values), 0)

def create_train_input_fn(path):
    def input_fn():    
        dataset = (
            tf.contrib.data.TextLineDataset(path)  # create a dataset from a file
                .filter(filter_empty_lines)  # ignore empty lines
                .map(csv_decoder)  # parse each row
                .shuffle(buffer_size=1000)  # shuffle the dataset
                .repeat()  # repeate indefinitely
                .batch(32)) # batch the data

        # create iterator
        columns = dataset.make_one_shot_iterator().get_next()
        
        # separate the label and convert it to true/false
        income = tf.equal(columns.pop('income')," >50K") 
        return columns, income
    return input_fn

def create_test_input_fn(path):
    def input_fn():    
        dataset = (
            tf.contrib.data.TextLineDataset(path)
                .skip(1) # The test file has a strange first line, we want to ignore this.
                .filter(filter_empty_lines)
                .map(csv_decoder)
                .batch(32))

        # create iterator
        columns = dataset.make_one_shot_iterator().get_next()
        
        # separate the label and convert it to true/false
        income = tf.equal(columns.pop('income')," >50K") 
        return columns, income
    return input_fn

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




