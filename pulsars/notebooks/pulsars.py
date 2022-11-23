# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: JKERNEL
#     language: python
#     name: jkernel
# ---

# # Pulsars detection in sample of candidates

# ## Introduction
# Pulsars are the class of neutron stars which have a strong electromagnetic field. The astrophysics simulations show that this field accelerates particles (mostly electrons and positrons) up to values close to the speed of light (https://www.youtube.com/watch?v=jwC6_oWwbSE). The part of the positrons cause strong gamma ray emission along the axis of magnetic poles. However, the star rotates around a diffetent fixed axis. Therefore, the beam of emission is pointing toward Earth only once each rotational period. 

# <img src="scheme.jpg" width="40%" style="float: left; margin-right: 25px;"/>
# <img src="pulsars.gif" width="55%" style="float: left;"/>

# We analyse the HTRU2 dataset given by Dr Robert Lyon and available at https://archive.ics.uci.edu/ml/datasets/HTRU2. The dataset consists of the target class and first 4 statistical moments observed for both integrated pulse profile and signal-to-noise ratio of object dispersion measure (DM-SNR). Examples of profile and DM curve of pulsar candidate PSR J1706-6118 are presented in the following figure (http://www.scienceguyrob.com/wp-content/uploads/2016/12/WhyArePulsarsHardToFind_Lyon_2016.pdf).

# <img src="IP_DM-SNR.jpg" width="80%" style="float: left; margin-right: 25px;"/>

# ## Data Overview

# The dataset is downloaded and available in the folder `../../data/pulsar_stars/`. We import the CSV file by means of Pandas 

import pandas as pd
data = pd.read_csv('../../data/pulsar_stars/pulsar_stars.csv')
data.head(5)

# Let us rename the columns to have more compact titles

columns = data.columns
column_names = list(['IP1', 'IP2', 'IP3', 'IP4', 'DM1', 'DM2', 'DM3', 'DM4'])
data = data.rename(columns = {
    columns[0]: column_names[0], columns[1]: column_names[1], columns[2]: column_names[2], columns[3]: column_names[3],
    columns[4]: column_names[4], columns[5]: column_names[5], columns[6]: column_names[6], columns[7]: column_names[7],
    columns[8]: 'target'
})
data.head(4)

# The general statistical description of the data is given by

data.describe()

# It can be seen that standard deviations of features are in range from `1.064040` to `106.514540` and all attributes have non-zero means. We scale the dataset to have unit variances and zero means because it is more convinient to work with.

# +
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = data.copy()
scaled_data[column_names] = scaler.fit_transform(data[column_names].to_numpy())
scaled_data.describe()
# -

# In order to see if there is dependence between features let us build the correlation matrix 

# +
import matplotlib.pyplot as plt
import seaborn as sns

palette = sns.light_palette("purple", reverse = True)

corr = scaled_data.filter(regex = "[^target]").corr()
plt.figure(figsize = (16, 7))
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, cmap = palette, square = True, annot = True)
# -

# There is a strong correlation between `IP3` - `IP4` and `DM3` - `DM4`. 

# +
import os
import itertools
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import urllib
import math
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# -

# Remove target class

scaled_data_without_target = scaled_data.filter(regex = "[^target]")
x_train, x_test, y_train, y_test = train_test_split(scaled_data_without_target.values, scaled_data.target.values,
                                                    test_size = 0.2, random_state = 5)

# Apply Random Forest classifier

rf = RandomForestClassifier(n_estimators = 200, random_state = 3)
rf.fit(x_train, y_train)
form = rf.score(x_test, y_test) * 100
print("Random Forrest accuracy : {0:.2f}%".format(form))

# Build confusion matrix

y_pred = rf.predict(scaled_data_without_target)
print(confusion_matrix(scaled_data.target.T.values, y_pred))

# Check classification report

print(classification_report(scaled_data.target.T.values, y_pred))

# Calculate ROC-curve


predict = rf.predict(x_test)
predict_probabilities = rf.predict_proba(x_test)
fpr, tpr, _ = roc_curve(y_test, predict_probabilities[:, :1])

lowest_prob = 0.9999
print(confusion_matrix(y_test, np.where(predict_probabilities[:, :1] > lowest_prob, 1, 0)))
print(classification_report(y_test, np.where(predict_probabilities[:, :1] > lowest_prob, 1, 0)))

# +
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predict_probabilities[:, :1])
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, color='red', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# -

thresholds


