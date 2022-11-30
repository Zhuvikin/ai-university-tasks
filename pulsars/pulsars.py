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

# +
import pandas as pd

data = pd.read_csv('../data/pulsar_stars/pulsar_stars.csv')
data.head(5)
# -

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

# %matplotlib inline

palette = sns.light_palette("purple", reverse = True)

corr = scaled_data.filter(regex = "[^target]").corr()
plt.figure(figsize = (16, 7))
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, cmap = palette, square = True, annot = True)
plt.show()
# -

# There is a strong correlation between `IP3` - `IP4` and `DM3` - `DM4`. However, not every feature is important for our classification issue. This is clearly seen from the following pairwise relationships. The purple points represent pulsars. The distribution densities are fit by histograms and ploted along the main diagnoal for each of the features.

# +
# palette2 = sns.color_palette(["#bbbbbb", "#a800a2"])

# pg = sns.PairGrid(scaled_data, palette = palette2, hue = "target", hue_order = [0, 1], vars = column_names)

# pg.map_diag(sns.kdeplot),
# pg.map_offdiag(plt.scatter, s = 2, alpha = 0.2)
# -

# It can be seen that some features allows one to split the data set linearly very well.

# Another important point is that our data is imbalanced. We should take it into account for some classification algorithms.

number_of_others, number_of_pulsars = scaled_data.target.value_counts()
scaled_data.target.value_counts().plot(kind = 'pie', labels = ['Others (' + str(number_of_others) + ')',
                                                               'Pulsars (' + str(number_of_pulsars) + ')'],
                                       figsize = (7, 7), colors = ['#e5e5e5', '#a800a2'])

# +
from sklearn.decomposition import PCA

groups = scaled_data.groupby(['target'])
nonPulsarsG = groups.get_group(0)
pulsarsG = groups.get_group(1)

pca = PCA(n_components = 2, random_state = 0)
pca.fit(scaled_data.filter(regex = "[^target]").values)
nonPulsarComponents = pca.transform(nonPulsarsG.filter(regex = "[^target]").values)
pulsarComponents = pca.transform(pulsarsG.filter(regex = "[^target]").values)

others_color = '#dddddd'
pulsars_color = '#a800a2'

fig = plt.figure(figsize = (10, 10))
ax1 = fig.add_subplot(111)
ax1.scatter(nonPulsarComponents[:, 0], nonPulsarComponents[:, 1], s = 10, c = others_color, label = 'Non-Pulsars')
ax1.scatter(pulsarComponents[:, 0], pulsarComponents[:, 1], s = 10, c = pulsars_color, label = 'Pulsars')
ax1.axis('tight')
plt.legend(loc = 'lower left');
plt.show()

# +
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

outlier_contamination = 0.1
pulsars_outlier_threshold = 0.3
non_pulsars_outlier_threshold = 0.5

fig = plt.figure(figsize = (16, 8))

clf = LocalOutlierFactor(n_neighbors = 20, contamination = outlier_contamination)
y_outlier_pred = clf.fit_predict(pulsarComponents)
X_scores = clf.negative_outlier_factor_
ax1 = fig.add_subplot(121)
ax1.scatter(pulsarComponents[:, 0], pulsarComponents[:, 1], color = pulsars_color, s = 3., label = 'Pulsars')
pulsars_radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
pulsars_radius[pulsars_radius < pulsars_outlier_threshold] = 0
ax1.scatter(pulsarComponents[:, 0], pulsarComponents[:, 1], s = 1000 * pulsars_radius, edgecolors = 'r',
            facecolors = 'none', label = 'Outliers')
ax1.axis('tight')
legend = ax1.legend(loc = 'lower left')
legend.legendHandles[1]._sizes = [20]

clf = LocalOutlierFactor(n_neighbors = 20, contamination = outlier_contamination)
y_outlier_pred = clf.fit_predict(nonPulsarComponents)
X_scores = clf.negative_outlier_factor_
ax1 = fig.add_subplot(122)
ax1.scatter(nonPulsarComponents[:, 0], nonPulsarComponents[:, 1], color = '#999999', s = 3., label = 'Non-Pulsars')
non_pulsars_radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
non_pulsars_radius[non_pulsars_radius < non_pulsars_outlier_threshold] = 0
ax1.scatter(nonPulsarComponents[:, 0], nonPulsarComponents[:, 1], s = 1000 * non_pulsars_radius, edgecolors = 'r',
            facecolors = 'none', label = 'Outliers')
ax1.axis('tight')
legend = ax1.legend(loc = 'lower left')
legend.legendHandles[1]._sizes = [20]

plt.show()

# +
remove_outliers = True

clear_data = scaled_data.copy()

if remove_outliers:
    outlier_positions = np.where(non_pulsars_radius > 0)[0]
    outlier_indexes = list(map(lambda position: nonPulsarsG.index[position], outlier_positions))
    nonPulsarsOutliers = nonPulsarsG.iloc[outlier_positions, :]
    nonPulsarsClear = nonPulsarsG.drop(outlier_indexes)

    outlier_positions = np.where(pulsars_radius > 0)[0]
    outlier_indexes = list(map(lambda position: pulsarsG.index[position], outlier_positions))
    pulsarsOutliers = pulsarsG.iloc[outlier_positions, :]
    pulsarsClear = pulsarsG.drop(outlier_indexes)

    clear_data = pd.concat([nonPulsarsClear, pulsarsClear]).sample(frac = 1, random_state = 1).reset_index(drop = True)
# -

# ## Classification Model

# We split our data to train and test sets

# +
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split

X = clear_data.filter(regex = "[^target]").values
y = clear_data.target.values
# -

# ### Method 1

# +
from benedict import benedict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score, classification_report

test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = 0)

# Add outliers to the test set
if remove_outliers:
    X_test = np.vstack(
        [nonPulsarsOutliers.filter(regex = "[^target]").values, pulsarsOutliers.filter(regex = "[^target]").values,
         X_test])
    y_test = np.hstack([nonPulsarsOutliers.target.values, pulsarsOutliers.target.values, y_test])

target_names = ['Non-pulsars', 'Pulsars']


def measure_test_performace(classifier):
    y_pred = classifier.predict(X_test)
    report = benedict(classification_report(y_test, y_pred, target_names = target_names, output_dict = True))

    probas = classifier.predict_proba(X)
    fpr, tpr, thresholds = roc_curve(y, probas[:, 1])
    report['auc'] = auc(fpr, tpr)
    report['fpr'] = fpr
    report['tpr'] = tpr
    report['predict_time'] = 1
    return report


# -

# ### Method 2

# Let resample our training data to have equal amount of pulsars and the other stars. We use undersampling technique for the further research as it shows better training results.

# +
# from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler

# sampler = ADASYN()
# X_oversampled, y_oversampled = sampler.fit_sample(X, y)

sampler = RandomUnderSampler(random_state = 1)
X_undersampled, y_undersampled = sampler.fit_sample(X, y)

print(X.shape)
# print(X_oversampled.shape)
print(X_undersampled.shape)
# -

# Implement classifier evaluation helper function using cross-validation and ROC analysis. We use Stratified K-Fold alghoritm in order to split our data for the training and validation


# +
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, auc, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from matplotlib.pyplot import figure
from functools import reduce
import time


def average_report(reports):
    result = benedict()
    for path in set(benedict(reports[0]).keypaths()).difference(
            list(['Non-pulsars', 'Pulsars', 'macro avg', 'weighted avg'])):
        average = reduce((lambda acc, report: acc + benedict(report)[path]), reports, 0) / len(reports)
        result[path] = average
    return result


def train_model(classifier, X_train, y_train, X_test, y_test, tprs, aucs, reports, mean_fpr):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Compute reports
    report = classification_report(y_test, y_pred, target_names = target_names, output_dict = True)
    # Compute ROC curve and area the curve
    probas = classifier.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    report['auc'] = roc_auc
    reports.append(report)


def measures_stratified_cv(classifier, X_set, y_set, mean_fpr):
    cv = StratifiedKFold(n_splits = 5, random_state = 0)
    tprs = []
    aucs = []
    reports = []
    i = 0
    for train, test in cv.split(X_set, y_set):
        train_model(classifier, X_set[train], y_set[train], X_set[test], y_set[test], tprs, aucs, reports, mean_fpr)
        i += 1
    result = dict()
    result['tprs'] = tprs
    result['aucs'] = aucs
    result['reports'] = reports
    return result


def evaluate_classifier(classifier, X_set = None, y_set = None, generate_cv_measures = None):
    fig = plt.figure(figsize = (17, 4))

    start = time.time()
    if generate_cv_measures is not None:
        mean_fpr = np.linspace(0, 1, 100)
        measures = generate_cv_measures(classifier, X_set, y_set, mean_fpr)
    else:
        classifier.fit(X_train, y_train)
    train_time = time.time() - start

    overall_report = measure_test_performace(classifier)

    ax1 = fig.add_subplot(121)
    plt.rcParams["figure.figsize"] = (5, 5)

    if generate_cv_measures is not None:
        tprs = measures['tprs']
        aucs = measures['aucs']
        reports = measures['reports']

        mean_tpr = np.mean(tprs, axis = 0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax1.plot(mean_fpr, mean_tpr, color = '#a800a2',
                 label = r'Mean (AUC = %0.4f $\pm$ %0.4f)' % (mean_auc, std_auc), lw = 2, alpha = 1)
    else:
        ax1.plot(overall_report['fpr'], overall_report['tpr'], color = '#a800a2',
                 label = r'Mean (AUC = %0.4f)' % (overall_report['auc']), lw = 2, alpha = 1)

    if generate_cv_measures is not None:
        std_tpr = np.std(tprs, axis = 0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color = 'grey', alpha = 0.2,
                         label = r'$\pm$ $\sigma$')

    ax1.set_aspect(1.0)
    plt.xlim([0, 0.25])
    plt.ylim([0.75, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of ' + classifier.__class__.__name__)
    plt.legend(loc = "lower right")

    ax2 = fig.add_subplot(122)
    font_size = 10
    ax2.axis('off')

    if generate_cv_measures is not None:
        report = average_report(reports)

        def rf(path):
            return "{0:.4f}".format(report[path])

    def gf(path):
        return "{0:.4f}".format(overall_report[path])

    if generate_cv_measures is not None:
        cell_text = [
            [rf('Non-pulsars.precision'), rf('Non-pulsars.recall'), rf('Non-pulsars.f1-score'),
             gf('Non-pulsars.precision'), gf('Non-pulsars.recall'), gf('Non-pulsars.f1-score')],

            [rf('Pulsars.precision'), rf('Pulsars.recall'), rf('Pulsars.f1-score'),
             gf('Pulsars.precision'), gf('Pulsars.recall'), gf('Pulsars.f1-score')],
            [rf('macro avg.precision'), rf('macro avg.recall'), rf('macro avg.f1-score'),
             gf('macro avg.precision'), gf('macro avg.recall'), gf('macro avg.f1-score')],
            [rf('weighted avg.precision'), rf('weighted avg.recall'), rf('weighted avg.f1-score'),
             gf('weighted avg.precision'), gf('weighted avg.recall'), gf('weighted avg.f1-score')]]
    else:
        cell_text = [
            [gf('Non-pulsars.precision'), gf('Non-pulsars.recall'), gf('Non-pulsars.f1-score')],
            [gf('Pulsars.precision'), gf('Pulsars.recall'), gf('Pulsars.f1-score')],
            [gf('macro avg.precision'), gf('macro avg.recall'), gf('macro avg.f1-score')],
            [gf('weighted avg.precision'), gf('weighted avg.recall'), gf('weighted avg.f1-score')]]

    rows_labels = ['Non-Pulsars', 'Pulsars', 'Macro Avg', 'Weighted Avg']

    if generate_cv_measures is not None:
        column_labels = ['CV Avg\nPrecision', 'CV Avg\nRecall', 'CV Avg\nF1-measure', 'Test\nPrecision', 'Test\nRecall',
                         'Test\nF1-measure']
    else:
        column_labels = ['Test\nPrecision', 'Test\nRecall', 'Test\nF1-measure']

    gray = '#dddddd'
    purple = '#eeddee'
    gray_l = '#efefef'
    purple_l = '#ffefff'

    colors = []
    if generate_cv_measures is not None:
        colors = [
            [gray_l, gray_l, gray_l, purple_l, purple_l, purple_l],
            [gray_l, gray_l, gray_l, purple_l, purple_l, purple_l],
            [gray, gray, gray, purple, purple, purple],
            [gray, gray, gray, purple, purple, purple],
        ]
    else:
        colors = [
            [purple_l, purple_l, purple_l],
            [purple_l, purple_l, purple_l],
            [purple, purple, purple],
            [purple, purple, purple],
        ]

    if generate_cv_measures is not None:
        foolter_cell_colors = [[gray_l, gray_l, purple_l, purple_l]]
        cellText = [[r'AUC: %0.4f' % mean_auc, 'Accuracy: ' + rf('accuracy'), r'AUC: %0.4f' % overall_report['auc'],
                     'Accuracy: ' + gf('accuracy')]]
    else:
        foolter_cell_colors = [[purple_l, purple_l]]
        cellText = [[r'AUC: %0.4f' % overall_report['auc'], 'Accuracy: ' + gf('accuracy')]]

    vertical_scaling = 2.5
    footer_2 = plt.table(cellText = [['Train time: %0.4f' % train_time, '2']],
                         colLabels = ['', ''],
                         loc = 'bottom',
                         cellColours = [['w', 'w']],
                         cellLoc = 'center',
                         bbox = [0, -0.1, 1, 0.26])

    footer_2.scale(1, vertical_scaling)
    footer_2.auto_set_font_size(False)
    footer_2.set_fontsize(font_size)

    footer = plt.table(cellText = cellText,
                       colLabels = ['', '', '', ''],
                       loc = 'bottom',
                       cellColours = foolter_cell_colors,
                       cellLoc = 'center',
                       bbox = [0, 0.025, 1, 0.26])

    footer.scale(1, vertical_scaling)
    footer.auto_set_font_size(False)
    footer.set_fontsize(font_size)

    table = ax2.table(cellText = cell_text, rowLabels = rows_labels, cellColours = colors,
                      colLabels = column_labels, loc = 'center', cellLoc = 'center')
    table.scale(1, vertical_scaling)
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    return overall_report


# -

# ### Method 3

# +
from math import floor, ceil
import random

train_number_of_others = floor((1 - test_size) * number_of_others)
train_number_of_pulsars = floor((1 - test_size) * number_of_pulsars)

non_pulsars_to_pulsars_ratio = floor(train_number_of_others / train_number_of_pulsars)

number_of_pulsars_folds = 5
number_of_non_pulsars_folds = number_of_pulsars_folds * non_pulsars_to_pulsars_ratio

groups = clear_data.groupby(['target'])
nonPulsars = groups.get_group(0)
pulsars = groups.get_group(1)

nonPulsars_X = nonPulsars.copy()
pulsars_X = pulsars.copy()

print('Number of non-pulsar stars is at least', non_pulsars_to_pulsars_ratio, 'times higher.')
new_number_of_others = number_of_non_pulsars_folds * floor(
    non_pulsars_to_pulsars_ratio * train_number_of_pulsars / number_of_non_pulsars_folds)
to_be_removed = train_number_of_others - new_number_of_others
print('Drop', '{0:.1f}'.format(100 * to_be_removed / train_number_of_others),
      '% of non-pulsar stars to have amount divisible by', non_pulsars_to_pulsars_ratio, 'and', number_of_pulsars_folds)

random.seed(5)
indexes_to_remove = list(
    map(lambda i: nonPulsars_X.index[i], np.array(random.sample(range(nonPulsars.shape[0]), to_be_removed))))

nonPulsars_X = nonPulsars_X.drop(indexes_to_remove)


# +
def split_dataframe_to_chunks(df, n):
    df_len = len(df)
    count = 0
    dfs = []

    while True:
        if count > df_len - 1:
            break

        start = count
        count += n
        dfs.append(df.iloc[start: count])
    return dfs


nonPulsarFolds = split_dataframe_to_chunks(nonPulsars_X, ceil(new_number_of_others / 45))
pulsarFolds = split_dataframe_to_chunks(pulsars_X, ceil(number_of_pulsars / 5))


# -

def measures_full_set_cv(classifier, nonPulsarFolds, pulsarFolds, mean_fpr):
    tprs = []
    aucs = []
    reports = []
    i = 0

    combinations = list(map(lambda k: [k, k % number_of_pulsars_folds], range(0, number_of_non_pulsars_folds)))
    for np_k, p_k in combinations:
        #         print(str(i))
        np_rest = list(range(0, number_of_non_pulsars_folds))
        np_rest.remove(np_k)

        p_rest = list(range(0, number_of_pulsars_folds))
        p_rest.remove(p_k)

        np_other_folds = pd.concat(list(map(lambda idx: nonPulsarFolds[idx], np_rest)))
        p_other_folds = pd.concat(list(map(lambda idx: pulsarFolds[idx], p_rest)))

        train_folds = pd.concat([np_other_folds, p_other_folds]).sample(frac = 1, random_state = 2).reset_index(
            drop = True)
        test_fold = pd.concat([nonPulsarFolds[np_k], pulsarFolds[p_k]]).sample(frac = 1, random_state = 2).reset_index(
            drop = True)

        train_X = train_folds.filter(regex = "[^target]").values
        train_y = train_folds.target.values

        test_X = test_fold.filter(regex = "[^target]").values
        test_y = test_fold.target.values
        train_model(classifier, train_X, train_y, test_X, test_y, tprs, aucs, reports, mean_fpr)
        i += 1

    result = dict()
    result['tprs'] = tprs
    result['aucs'] = aucs
    result['reports'] = reports
    return result


# ### Logistic Regression Classifier

# +
from sklearn.linear_model import LogisticRegression

classifier_lr_1 = LogisticRegression(solver = 'lbfgs', random_state = 0, class_weight = 'balanced')
report_lr_1 = evaluate_classifier(classifier_lr_1)

classifier_lr_2 = LogisticRegression(solver = 'lbfgs', random_state = 0, class_weight = 'balanced')
report_lr_2 = evaluate_classifier(classifier_lr_2, X_undersampled, y_undersampled, measures_stratified_cv)

classifier_lr_3 = LogisticRegression(solver = 'lbfgs', random_state = 0, class_weight = 'balanced')
report_lr_3 = evaluate_classifier(classifier_lr_3, nonPulsarFolds, pulsarFolds, measures_full_set_cv)
# -

# ### C-Support Vector Classifier

# +
# import matplotlib.pyplot as plt
# from sklearn import svm

# classifier_svc_u = svm.SVC(kernel = 'linear', probability = True, random_state = 0, class_weight = 'balanced')
# avg_report_svc_u = evaluate_classifier(classifier_svc_u, X_undersampled, y_undersampled, measures_stratified_cv)

# classifier_svc_f = svm.SVC(kernel = 'linear', probability = True, random_state = 0, class_weight = 'balanced')
# avg_report_svc_f = evaluate_classifier(classifier_svc_f, nonPulsarFolds, pulsarFolds, measures_full_set_cv)
# -

# ### K-Neighbors Classifier

# +
# from sklearn.neighbors import KNeighborsClassifier

# classifier = KNeighborsClassifier(n_neighbors = 13)
# avg_report_kn = evaluate_classifier(classifier, X_undersampled, y_undersampled, measures_stratified_cv)
# -

# ### Decision Tree Classifier

# +
# from sklearn.tree import DecisionTreeClassifier

# classifier = DecisionTreeClassifier(random_state = 0)
# avg_report_dt = evaluate_classifier(classifier, X_undersampled, y_undersampled, measures_stratified_cv)
# -

# ### Random Forest Classifier

# +
# from sklearn.ensemble import RandomForestClassifier

# classifier = RandomForestClassifier(n_estimators = 200, random_state = 0)
# avg_report_rf = evaluate_classifier(classifier, X_undersampled, y_undersampled, measures_stratified_cv)
# -

# ### Naive Bayes Classifier

# +
# from sklearn.naive_bayes import GaussianNB

# classifier = GaussianNB()
# avg_report_nb = evaluate_classifier(classifier, X_undersampled, y_undersampled, measures_stratified_cv)
# -

# ### Gradient Boosting Classifier

# +
# from sklearn.ensemble import GradientBoostingClassifier

# classifier = GradientBoostingClassifier(n_estimators = 150, learning_rate = 0.1, max_features = 2,
#                                         max_depth = 2, random_state = 0)
# avg_report_gb = evaluate_classifier(classifier, X_undersampled, y_undersampled, measures_stratified_cv)
# -

# ### Neural Network Classifier

# +
# from sklearn.neural_network import MLPClassifier

# classifier = MLPClassifier(hidden_layer_sizes = (8, 4, 2), max_iter = 500, alpha = 0.00025,
#                            solver = 'adam', verbose = 0, random_state = 21, tol = 0.000000001, activation = 'relu')
# avg_report_mlp = evaluate_classifier(classifier, X_undersampled, y_undersampled, measures_stratified_cv)
# -

# ## Comparative Analysis

# Let us summarize the results given by the various alghoritms

# +
# from IPython.display import HTML, display
# import tabulate

# reports = [avg_report_svc_u, avg_report_lr_u, avg_report_kn_u, avg_report_dt_u,
#            avg_report_rf_u, avg_report_nb_u, avg_report_gb_u, avg_report_mlp_u]

# classifier_names = ['C-Support Vector', 'Logistic Regression', 'K-Neighbors', 'Decision Tree',
#                     'Random Forest', 'Naive Bayes', 'Gradient Boosting', 'Neural Network']

# measures_names = ['Accuracy', 'Precision', 'Recall', 'F1-measure', 'AUC']


# def cell(report, name):
#     if name == 'auc':
#         return "{0:.4f}".format(report[name])
#     else:
#         return "{0:.4f}".format(report[name] * 100) + ' %'


# def row(reports, name):
#     return list(map(lambda report: cell(report, name), reports))


# accuracies = row(reports, 'accuracy')
# precisions = row(reports, 'Pulsars.precision')
# recalls = row(reports, 'Pulsars.recall')
# f1Measures = row(reports, 'Pulsars.f1-score')
# AUCs = row(reports, 'auc')

# c_results = pd.DataFrame([
#     [measures_names[0]] + accuracies,
#     [measures_names[1]] + precisions,
#     [measures_names[2]] + recalls,
#     [measures_names[3]] + f1Measures,
#     [measures_names[4]] + AUCs
# ], columns = [''] + classifier_names)

# c_results.style.applymap(lambda x: 'background-color : magenta; color: white'
# if x == max(accuracies)
#    or x == max(precisions)
#    or x == max(recalls)
#    or x == max(f1Measures)
#    or x == max(AUCs) else '')
# -

# The greatest values of correspondent measures are highlighted. It can be seen that the best ML alghorithm for pulsars classification depends on what measure is going to be used

# ## Conclusion

# Various ML algorithms for the task of pulsars detection are evaluated. The source is re-scaled and re-balanced to have zero mean, unit variance and the same amount of elements in both target classes. We evaluated C-Support Vector, Logistic Regression, K-Neighbors, Decision Tree, Random Forest, Naive Bayes, Gradient Boosting and Neural Network classifiers in order to determine the best one. The general discriminitive ability is given by the area under ROC curve. The alghoritm based on neural network shows the best result by AUC measure. However, it is more inportant to have higher recall, F1-measure. So, Random Forest and Gradient Boosting do the job better.

# +
# classifier = svm.SVC(kernel = 'linear', probability = True, random_state = 0)
# avg_report_svc = evaluate_classifier(classifier, nonPulsarFolds, pulsarFolds, measures_full_set_cv)
# -

classifier_lr_f = LogisticRegression(solver = 'lbfgs', random_state = 0, class_weight = 'balanced')
avg_report_lr_f = evaluate_classifier(classifier_lr_f, nonPulsarFolds, pulsarFolds, measures_full_set_cv)

# +
# X = scaled_data.filter(regex = "[^target]").values
# y = scaled_data.target.values

# global_y_pred = classifier.predict(X)
# global_report = classification_report(y, y_pred, target_names = target_names, output_dict = True)

# print(global_report)

# +
# classifier = KNeighborsClassifier(n_neighbors = 13)
# avg_report_kn = evaluate_classifier(classifier, nonPulsarFolds, pulsarFolds, measures_full_set_cv)

# +
# classifier = DecisionTreeClassifier(random_state = 0)
# avg_report_dt = evaluate_classifier(classifier, nonPulsarFolds, pulsarFolds, measures_full_set_cv)

# +
# classifier = RandomForestClassifier(n_estimators = 200, random_state = 0)
# avg_report_rf = evaluate_classifier(classifier, nonPulsarFolds, pulsarFolds, measures_full_set_cv)

# +
# classifier = GaussianNB()
# avg_report_nb = evaluate_classifier(classifier, nonPulsarFolds, pulsarFolds, measures_full_set_cv)

# +
# classifier = GradientBoostingClassifier(n_estimators = 150, learning_rate = 0.1, max_features = 2,
#                                         max_depth = 2, random_state = 0)
# avg_report_gb = evaluate_classifier(classifier, nonPulsarFolds, pulsarFolds, measures_full_set_cv)

# +
# classifier = MLPClassifier(hidden_layer_sizes = (8, 4, 2), max_iter = 500, alpha = 0.00025,
#                            solver = 'adam', verbose = 0, random_state = 21, tol = 0.000000001, activation = 'relu')
# avg_report_mlp = evaluate_classifier(classifier, nonPulsarFolds, pulsarFolds, measures_full_set_cv)
# -
