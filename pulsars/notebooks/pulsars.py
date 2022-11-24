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

# There is a strong correlation between `IP3` - `IP4` and `DM3` - `DM4`. However, not every feature is important for our classification issue. This is clearly seen from the following pairwise relationships. The purple points represent pulsars. The distribution densities are fit by histograms and ploted along the main diagnoal for each of the features.

# +
palette2 = sns.color_palette(["#bbbbbb", "#a800a2"])

pg = sns.PairGrid(scaled_data, palette = palette2, hue = "target", hue_order = [0, 1], vars = column_names)

pg.map_diag(sns.kdeplot),
pg.map_offdiag(plt.scatter, s = 2, alpha = 0.2)
# -

# It can be seen that some features allows one to split the data set linearly very well.

# Another important point is that our data is imbalanced. We should take it into account for some classification algorithms.

number_of_others, number_of_pulsars = scaled_data.target.value_counts()
scaled_data.target.value_counts().plot(kind = 'pie', labels = ['Others (' + str(number_of_others) + ')',
                                                               'Pulsars (' + str(number_of_pulsars) + ')'],
                                       figsize = (7, 7), colors = ['#e5e5e5', '#a800a2'])

# ## Classification Model

# We split our data to train and test sets

# +
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split

X = scaled_data.filter(regex = "[^target]").values
y = scaled_data.target.values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
# -

# Implement classifier evaluation helper function using cross-validation and ROC analysis

# +
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, auc, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from matplotlib.pyplot import figure
from functools import reduce
from benedict import benedict


def average_report(reports):
    result = benedict()
    for path in set(benedict(reports[0]).keypaths()).difference(
            list(['Non-pulsars', 'Pulsars', 'macro avg', 'weighted avg'])):
        average = reduce((lambda acc, report: acc + benedict(report)[path]), reports, 0) / len(reports)
        result[path] = average
    return result


def evaluate_classifier(classifier):
    fig = plt.figure(figsize = (14, 4))
    target_names = ['Non-pulsars', 'Pulsars']

    cv = StratifiedKFold(n_splits = 6)
    tprs = []
    aucs = []
    f1_scores = []
    reports = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        classifier.fit(X[train], y[train])
        y_pred = classifier.predict(X[test])
        # Compute reports
        report = classification_report(y[test], y_pred, target_names = target_names, output_dict = True)
        reports.append(report)
        # Compute ROC curve and area the curve
        probas = classifier.predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        i += 1

    mean_tpr = np.mean(tprs, axis = 0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    ax1 = fig.add_subplot(121)
    plt.rcParams["figure.figsize"] = (5, 5)
    ax1.plot(mean_fpr, mean_tpr, color = '#a800a2',
             label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw = 2, alpha = 1)

    std_tpr = np.std(tprs, axis = 0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color = 'grey', alpha = 0.2,
                     label = r'$\pm$ 1 std. dev.')
    ax1.set_aspect(1.0)
    plt.xlim([0, 0.25])
    plt.ylim([0.75, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of ' + classifier.__class__.__name__)
    plt.legend(loc = "lower right")

    ax2 = fig.add_subplot(122)
    font_size = 14
    bbox = [0, 0, 1, 1]
    ax2.axis('off')

    report = average_report(reports)

    def f(path):
        return "{0:.4f}".format(report[path])

    cell_text = [
        [f('Non-pulsars.precision'), f('Non-pulsars.recall'), f('Non-pulsars.f1-score')],
        [f('Pulsars.precision'), f('Pulsars.recall'), f('Pulsars.f1-score')],
        ['', '', ''],
        [f('macro avg.precision'), f('macro avg.recall'), f('macro avg.f1-score')],
        [f('weighted avg.precision'), f('weighted avg.recall'), f('weighted avg.f1-score')],
        ['', '', ''],
        ['', '', f('accuracy')]]

    rows_labels = ['Non-pulsars', 'Pulsars', '', 'macro avg', 'weighted avg', '', 'accuracy']
    column_labels = ['avg precision', 'avg recall', 'avg f1-measure']

    table = ax2.table(cellText = cell_text, rowLabels = rows_labels,
                      colLabels = column_labels, loc = 'center')
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    return report


# -

# ### C-Support Vector Classifier

# +
from sklearn import svm

classifier = svm.SVC(kernel = 'linear', probability = True, random_state = 1)
avg_report_svc = evaluate_classifier(classifier)
# -

# ### Logistic Regression Classifier

# +
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(solver = 'lbfgs')
avg_report_lr = evaluate_classifier(classifier)
# -

# ### K-Neighbors Classifier

# +
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 13)
avg_report_kn = evaluate_classifier(classifier)
# -

# ### Decision Tree Classifier

# +
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
avg_report_dt = evaluate_classifier(classifier)
# -

# ### Random Forest Classifier

# +
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 200, random_state = 3)
avg_report_rf = evaluate_classifier(classifier)
# -

# ### Naive Bayes Classifier

# +
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
avg_report_nb = evaluate_classifier(classifier)
# -

# ### Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier

# Learning rate: 0.05

classifier = GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.05, max_features = 2,
                                        max_depth = 2, random_state = 0)
avg_report_gb_lr_005 = evaluate_classifier(classifier)

# Learning rate: 0.1

classifier = GradientBoostingClassifier(n_estimators = 150, learning_rate = 0.1, max_features = 2,
                                        max_depth = 2, random_state = 0)
avg_report_gb_lr_010 = evaluate_classifier(classifier)

# Learning rate: 0.2

classifier = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.2, max_features = 2,
                                        max_depth = 2, random_state = 0)
avg_report_gb_lr_020 = evaluate_classifier(classifier)

# Learning rate: 0.3

classifier = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.3, max_features = 2,
                                        max_depth = 2, random_state = 0)
avg_report_gb_lr_030 = evaluate_classifier(classifier)

# Learning rate: 0.4

classifier = GradientBoostingClassifier(n_estimators = 30, learning_rate = 0.4, max_features = 2,
                                        max_depth = 2, random_state = 0)
avg_report_gb_lr_040 = evaluate_classifier(classifier)

# Looks like learning rate equal to `0.1` does the job better

# ## Neural Network Classifiers

# ### MLP Classifier

# +
from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(hidden_layer_sizes=(3, 3, 3), max_iter=500, alpha=0.0001,
                           solver='sgd', verbose=10,  random_state=21, tol=0.000000001)
avg_report_mlp = evaluate_classifier(classifier)
# + {}
from sklearn.neural_network import MLPClassifier

classifier = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=500, alpha=0.0001,
                           solver='sgd', verbose=10,  random_state=21, tol=0.000000001)
avg_report_mlp = evaluate_classifier(classifier)
# -



