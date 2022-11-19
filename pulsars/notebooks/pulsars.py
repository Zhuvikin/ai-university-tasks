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

# # Pulsars

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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# -

# Import data set

data = pd.read_csv('~/workspace/ai-university/data/pulsar_stars/pulsar_stars.csv')

columns = data.columns
data = data.rename(columns={
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

scaled_data_without_target = scaled_data.filter(regex="[^target]")
x_train, x_test, y_train, y_test = train_test_split(scaled_data_without_target.values, scaled_data.target.values,
                                                    test_size=0.2, random_state=5)

# Apply Random Forest classifier

rf = RandomForestClassifier(n_estimators=200, random_state=3)
rf.fit(x_train, y_train)
form = rf.score(x_test, y_test) * 100
print("Random Forrest accuracy : {0:.2f}%".format(form))

# Check the confusion matrix

y_pred = rf.predict(scaled_data_without_target)
con_mat = confusion_matrix(scaled_data.target.T.values, y_pred)
print("Confusion matrix: ", con_mat)
