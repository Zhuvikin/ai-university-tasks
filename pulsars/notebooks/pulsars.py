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

# Pulsars

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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# -

data = pd.read_csv('../../data/pulsar_stars/pulsar_stars.csv')

# +
columns = data.columns
IP1, IP2, IP3, IP4 = columns[0], columns[1], columns[2], columns[3]
DM1, DM2, DM3, DM4 = columns[4], columns[5], columns[6], columns[7]

data = data.rename(columns={ 
    columns[0]: 'IP1', columns[1]: 'IP2', columns[2]: 'IP3', columns[3]: 'IP4',
    columns[4]: 'DM1', columns[5]: 'DM2', columns[6]: 'DM3', columns[7]: 'DM4',
    columns[8]: 'target'
})
# -


