# %% [code] {"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import copy

import keras
import csv
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, LSTM, Dense, concatenate
from keras.models import Model
from keras.preprocessing import sequence
from sklearn import preprocessing
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code] {"jupyter":{"outputs_hidden":false}}
#This is an edit
#Useful links: https://keras.io/getting-started/functional-api-guide/

# %% [code] {"jupyter":{"outputs_hidden":false}}
# import train and test
df_train = pd.read_csv('/kaggle/input/cs446-fa19/train.csv')
df_test = pd.read_csv('/kaggle/input/cs446-fa19/test.csv')

# %% [code] {"jupyter":{"outputs_hidden":false}}
def drop_features(df):
    df = df.drop(columns=['optimizer'])
    df = df.drop(columns=['init_params_mu'])
    df = df.drop(columns=['init_params_std'])
    df = df.drop(columns=['init_params_l2'])
    df = df.drop(columns=['criterion'])
    df = df.drop(columns=['batch_size_test'])
    df = df.drop(columns=['batch_size_val'])
    df = df.drop(columns=['batch_size_train'])
    df = df.drop(columns=['id'])
    df = df.drop(df.columns[0], axis=1)
    return df

# %% [code] {"jupyter":{"outputs_hidden":false}}
def transform_single_arch_and_hp(arch_and_hp):
    layers = arch_and_hp.split(':')
    for idx, layer in enumerate(layers):
        layers[idx]= layer.split('(')
    for idx, layer in enumerate(layers):
        layers[idx]= layer[-1][:-1]
    layers = layers[:-1]
    return ' '.join(layers)
def transform_arch_and_hp(df):
    df['arch_and_hp'] = df['arch_and_hp'].apply(transform_single_arch_and_hp)
    return df
def drop_val_loss_and_train_loss(df):
    df = df.drop(columns=['val_loss'])
    df = df.drop(columns=['train_loss'])
    return df

# %% [code] {"jupyter":{"outputs_hidden":false}}
def transform_features(df):
    df = drop_features(df)
    df = transform_arch_and_hp(df)
    return df

# %% [code] {"jupyter":{"outputs_hidden":false}}
df_train = transform_features(df_train)
df_test = transform_features(df_test)
df_train = drop_val_loss_and_train_loss(df_train)

# %% [code] {"jupyter":{"outputs_hidden":false}}
archs_and_hps = []
for arch in df_train['arch_and_hp']:
    archs_and_hps.append(arch)

# %% [code] {"jupyter":{"outputs_hidden":false}}
np.unique(np.array(archs_and_hps)).size

# %% [code] {"jupyter":{"outputs_hidden":false}}
t = Tokenizer(oov_token='oov')
t.fit_on_texts(archs_and_hps)

# %% [code] {"jupyter":{"outputs_hidden":false}}
# print(t.word_counts)
# print(t.document_count)
# print(t.word_index)
# print(t.word_docs)
num_words = len(t.word_index)
print(num_words)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def normalize(tokenizer, df_train, df_test):
    # First do df_train
    train_labels = df_train[['val_error', 'train_error']]
    df_train = df_train.drop(['val_error', 'train_error'], axis=1)
    train_arch = tokenizer.texts_to_sequences(df_train['arch_and_hp'])
    train_data = np.float32(df_train.drop('arch_and_hp', axis=1))
#     print(train_data)
    scaler = preprocessing.StandardScaler()
    train_data = scaler.fit_transform(train_data)
    
    # Now do df_test
    test_arch = tokenizer.texts_to_sequences(df_test['arch_and_hp'])
    test_data = np.float32(df_test.drop('arch_and_hp', axis=1))
    test_data = scaler.transform(test_data)
    return (train_arch, train_data, train_labels, test_arch, test_data)

# %% [code] {"jupyter":{"outputs_hidden":false}}
train_arch, train_data, train_labels, test_arch, test_data = normalize(t, df_train, df_test)

# %% [code] {"jupyter":{"outputs_hidden":false}}
max_num_layers = len(max(train_arch,key=len))
print(max_num_layers)

# %% [code] {"jupyter":{"outputs_hidden":false}}
train_arch = sequence.pad_sequences(train_arch, maxlen=max_num_layers)
test_arch = sequence.pad_sequences(test_arch, maxlen=max_num_layers)
print(type(test_arch[0,0]))

# %% [code] {"jupyter":{"outputs_hidden":false}}
print(train_data[:,1].size)
print(np.mean(train_data[:,2]))
print(np.std(train_data[:,2]))

# %% [code] {"jupyter":{"outputs_hidden":false}}
print(train_data.shape)

# %% [code] {"jupyter":{"outputs_hidden":false}}
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

# %% [code] {"jupyter":{"outputs_hidden":false}}
data_size = train_data.shape[1]
arch_input = Input(shape=(max_num_layers,), dtype='int32', name='arch_input')
x = Embedding(output_dim=4, input_dim=num_words+1, input_length=max_num_layers)(arch_input)
lstm_out = LSTM(32)(x)

data_input = Input(shape=(data_size,), name='data_input')
x = keras.layers.concatenate([lstm_out, data_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

val_error_output = Dense(1, activation='sigmoid', name='val_error_output')(x)
train_error_output = Dense(1, activation='sigmoid', name='train_error_output')(x)

model = Model(inputs=[arch_input, data_input], outputs=[val_error_output, train_error_output])
model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=[coeff_determination], loss_weights=[0.5, 0.5])
model.fit({'arch_input': train_arch, 'data_input': train_data},
          {'val_error_output': train_labels['val_error'].data, 'train_error_output': train_labels['train_error'].data},
          epochs=50, batch_size=32)

# %% [code] {"jupyter":{"outputs_hidden":false}}
test_val_error, test_train_error = model.predict({'arch_input': test_arch, 'data_input': test_data})
num_tests = test_val_error.size

out_csv = []
with open('results.csv', mode='w') as results:
    results_writer = csv.writer(results, delimiter=',', quotechar='"')
    results_writer.writerow(['id', 'predicted'])
    for i in range(test_val_error.size):
        results_writer.writerow(['test_' + str(i) + '_val_error', test_val_error[i][0]])
        results_writer.writerow(['test_' + str(i) + '_train_error', test_train_error[i][0]])