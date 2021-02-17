# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:12:50 2021

@author: Zirak
"""

import dask.bag as db
import dask.array as da
import dask.dataframe as df
import math
import pandas as pd
import numpy as np
import requests
from dask_ml import naive_bayes





from dask.distributed import Client
client = Client()  # set up local cluster on your laptop
client

def display_object_method(obj):
    object_methods = [method_name for method_name in dir(obj)
                  if callable(getattr(obj, method_name))]
    print(object_methods)


def get_split_words(word_bag):    
    split_words = word_bag.str.split()
    stripped_words = split_words.flatten().map(lambda x: x.strip())
    word_array = stripped_words.filter(lambda x: x!= "")
    return word_array

def get_filtered_words(words):
    stopwords = open("C:/Users/Zirak/Project1/data/stopwords.txt", "r").read().split("\n")
    return words.filter(lambda x: x not in stopwords)



def get_words_for_document(document):
    word_bag = db.read_text(document)
    split_words = get_split_words(word_bag)
    long_words = split_words.filter(lambda x: len(x) < 3)
#     words_without_punctuations = long_words.map(puntuation_stripper)
    return get_filtered_words(long_words)    
#     return long_words

def get_IDF(words_by_document):
    unique_words = []
    for words_for_single_document in words_by_document:
        unique_words.append(words_for_single_document.distinct())
    large_bag = db.concat(unique_words)
    frequencies = large_bag.frequencies()
    idf = frequencies.map(lambda x: (x[0], round(math.log((len(words_by_document) + 1)/x[1]), 10)))
    return idf




def get_TF_IDF(tf, idf):
    tf = np.asarray(tf.frequencies().compute())
    tf_idf = {}
    for idf_item in idf: 
        col_name = str(int(idf_item[0], 16))
        tf_idf[col_name] = idf_item[1]
        for i in range(len(tf)):
            tf_item = tf[i]
            if idf_item[0] == tf_item[0]:
                tf_idf[col_name] = float(idf_item[1]) * float(tf_item[1])
                np.delete(tf, i)
                break
    return tf_idf

set_name = "X_small_train.txt"
# set_name = "X_train.txt"
# url = "data/train/" + set_name
url = 'https://storage.googleapis.com/uga-dsp/project1/files/' + set_name
file_names = pd.read_csv(url, header=None)[0].to_numpy()



all_file_words = []
for file_name in file_names:
#     file_name = "data/train/" + file_name + ".bytes"
    file_name = "https://storage.googleapis.com/uga-dsp/project1/data/bytes/" + file_name + ".bytes"
    
    all_file_words.append(get_words_for_document(file_name))
idf = get_IDF(all_file_words)


idf = np.asarray(idf.topk(256, key=0).compute())


cols = np.arange(255, -1, -1).astype(str)
pd_df = pd.DataFrame(columns=cols)

for file_words in all_file_words:
    tf_idf_bag_array = get_TF_IDF(file_words, idf)
    pd_df = pd_df.append(tf_idf_bag_array, ignore_index=True)
    
    


csv_name = "C:/Users/Zirak/Project1/data/" + set_name + ".csv"
pd_df.to_csv(csv_name)

tf_idf_df = df.from_pandas(pd_df, npartitions=1)
final_dataframe = tf_idf_df.compute()


final_dataframe



#Naive Bayes Experimentation

X_small_data = df.read_csv("C:/Users/Zirak/Project1/data/X_small_train.txt.csv")

set_name1 = "y_small_train.txt"
# set_name = "X_train.txt"
# url = "data/train/" + set_name
url = 'https://storage.googleapis.com/uga-dsp/project1/files/' + set_name1
y_small_labels = pd.read_csv(url, header=None)[0].to_numpy().reshape(-1,1)


#from dask_ml.preprocessing import OneHotEncoder

#encoder = OneHotEncoder(sparse=True)
#y_small_labels_onehot = encoder.fit(y_small_labels)


model = naive_bayes.GaussianNB()

#model.fit(X_small_data, y_small_labels)
model.fit(final_dataframe, y_small_labels)


#Setting Prediction data/test data
set_name2 = "X_small_test.txt"
url = 'https://storage.googleapis.com/uga-dsp/project1/files/' + set_name2
file_name_x_test = pd.read_csv(url, header=None)[0].to_numpy()






all_file_words_x_test = []
for file_name in file_name_x_test:
#     file_name = "data/train/" + file_name + ".bytes"
    file_name = "https://storage.googleapis.com/uga-dsp/project1/data/bytes/" + file_name + ".bytes"
    
    all_file_words_x_test.append(get_words_for_document(file_name))
idf_x_test = get_IDF(all_file_words_x_test)


idf_x_test = np.asarray(idf_x_test.topk(256, key=0).compute())


cols = np.arange(255, -1, -1).astype(str)
pd_df_x_test = pd.DataFrame(columns=cols)

for file_words in all_file_words_x_test:
    tf_idf_bag_array_test = get_TF_IDF(file_words, idf_x_test)
    pd_df_x_test = pd_df_x_test.append(tf_idf_bag_array_test, ignore_index=True)
  
set_name_test = "X_small_test"    
csv_name_test = "C:/Users/Zirak/Project1/data/" + set_name_test + ".csv"
pd_df_x_test.to_csv(csv_name_test)   


prediction = model.predict(pd_df_x_test)