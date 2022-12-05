import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.externals import joblib
seaborn.set_style('whitegrid')

label = preprocessing.LabelEncoder()


def normalize(dataframe):
    return (dataframe - dataframe.min()) / dataframe.max()


dataframe = pandas.read_csv("table.csv")
data_x = dataframe.iloc[:,1]
data_y = dataframe.iloc[:,2]

data_x = normalize(data_x)
data_y = label.fit_transform(data_y)


data_x = numpy.array(data_x).reshape(-1,1)
data_y = numpy.array(data_y)


def visualization(data_x, data_y):
    y = label_binarize(data_y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    learn_x, test_x, learn_y, test_y = train_test_split(data_x, y, test_size=.33, random_state=0)
    classifier = RandomForestClassifier(n_estimators=200, max_depth=4, criterion='gini', min_samples_leaf=3)
    count_y = classifier.fit(learn_x, learn_y).predict(test_x)
    a = dict()
    b = dict()
    roc_auc = dict()
    for i in range(n_classes):
        a[i], b[i], _ = roc_curve(test_y[:, i], count_y[:, i])
        roc_auc[i] = auc(a[i], b[i])
    a["micro"], b["micro"], _ = roc_curve(test_y.ravel(), count_y.ravel())
    roc_auc["micro"] = auc(a["micro"], b["micro"])

    all_a = numpy.unique(numpy.concatenate([a[i] for i in range(n_classes)]))
    mean_b = numpy.zeros_like(all_a)
    for i in range(n_classes):
        mean_b += interp(all_a, a[i], b[i])
    mean_b /= n_classes

    a["macro"] = all_a
    b["macro"] = mean_b
    roc_auc["macro"] = auc(a["macro"], b["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(a["micro"], b["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(a["macro"], b["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(a[i], b[i], color=color,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.ylim([0.0, 1.05])
    plt.xlim([-0.05, 1.0])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()


def save_model(data_x, data_y):
    y = label_binarize(data_y, classes=[0, 1, 2])

    learn_x, test_x, learn_y, test_y = train_test_split(data_x, y, test_size=.33, random_state=0)
    classifier = RandomForestClassifier(n_estimators=200, max_depth=4, criterion='gini', min_samples_leaf=3)
    model = classifier.fit(learn_x, learn_y)
    joblib.dump(model, 'model.pkl')


def make_prediction(score):
    model = joblib.load('model.pkl')
    data = numpy.array(score).reshape(-1, 1)
    prediction = model.predict(data)
    return prediction