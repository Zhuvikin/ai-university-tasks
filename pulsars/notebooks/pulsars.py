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
# Pulsars are the class of neutron stars which have a strong electromagnetic field. The astrophysics simulations show that this field accelerates particles (mostly electrons and positrons) up to values close to the speed of light [refer]. The part of the positrons cause strong gamma ray emission along the axis of magnetic poles. However, the star rotates around a diffetent fixed axis. Therefore, the beam of emission is pointing toward Earth only once each rotational period. 

# <img src="scheme.jpg" width="40%" style="float: left; margin-right: 25px;"/>
# <img src="pulsars.gif" width="55%" style="float: left;"/>

# We analyse the HTRU2 dataset given by Dr Robert Lyon and available at https://archive.ics.uci.edu/ml/datasets/HTRU2. The dataset consists of the target class and first 4 statistical moments observed for both integrated pulse profile and signal-to-noise ratio of object dispersion measure (DM-SNR). Examples of profile and DM curve of pulsar candidate PSR J1706-6118 is presented in the following figure (http://www.scienceguyrob.com/wp-content/uploads/2016/12/WhyArePulsarsHardToFind_Lyon_2016.pdf).

# <img src="IP_DM-SNR.jpg" width="80%" style="float: left; margin-right: 25px;"/>



# +
import pandas as pd
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
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

# Import data set

data = pd.read_csv('~/workspace/ai-university/data/pulsar_stars/pulsar_stars.csv')

columns = data.columns
data = data.rename(columns = {
    columns[0]: 'IP1', columns[1]: 'IP2', columns[2]: 'IP3', columns[3]: 'IP4',
    columns[4]: 'DM1', columns[5]: 'DM2', columns[6]: 'DM3', columns[7]: 'DM4',
    columns[8]: 'target'
})

data.describe()

# Standardize features to have zero mean and unit variance

scaler = StandardScaler()
scaled_data = data.copy()
scaled_data[['IP1', 'IP2', 'IP3', 'IP4', 'DM1', 'DM2', 'DM3', 'DM4']] = scaler.fit_transform(
    data[['IP1', 'IP2', 'IP3', 'IP4', 'DM1', 'DM2', 'DM3', 'DM4']].to_numpy())

scaled_data.describe()

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


