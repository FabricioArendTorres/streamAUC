"""
Example on how to use the library for tracking metrics.
For simplicity, sklearn is used as example model, with the iris dataset.

A large dataset at test time is simulated by resampling...

The fitting of the iris data is taken from the sklearn examples to
multiclass ROC, see:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html


For running this, you need to additionally install scikit-learn, tqdm,
and matplotlib (or just everything in the requirements-test.txt).

"""
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer

from tqdm import tqdm
import matplotlib.pyplot as plt

from streamauc import StreamingMetrics, AggregationMethod, auc
from streamauc import metrics

np.random.seed(1234)

##############################################################################
# setup data
##############################################################################
iris = load_iris()
X, y = iris.data, iris.target_names[iris.target]

random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.concatenate(
    [X, random_state.randn(n_samples, 200 * n_features)], axis=1
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
classifier = LogisticRegression(max_iter=1000)
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

label_binarizer = LabelBinarizer().fit(y_train)
y_test = np.argmax(label_binarizer.transform(y_test), -1)

### make test dataset absurdely large
idx = np.arange(y_test.shape[0])
idx_resampled = np.random.choice(idx, 100_000)
y_test = y_test[idx_resampled]
X_test = X_test[idx_resampled]
X_test += np.random.randn(*X_test.shape) * 0.1

##############################################################################
# simulate minibatches at test time
##############################################################################

# Select the number of thresholds for which we want to keep track of results.
stream_metrics = StreamingMetrics(
    num_thresholds=100,
    num_classes=3,
)

mb_size = 10_000
num_mbs = X_test.shape[0] // mb_size
for i in tqdm(range(num_mbs)):
    mb_X = X_test[mb_size * i:mb_size * (i + 1)]
    mb_y = y_test[mb_size * i:mb_size * (i + 1)]

    y_pred = classifier.predict_proba(mb_X)
    # remove check inputs for faster updates
    stream_metrics.update(y_true=mb_y, y_score=y_pred, check_inputs=False)


# ###### METRICS
# get AUC for any combination of metrics with
# micro, macro or no aggregation (i.e. onevsall).
_auc_onevsall = stream_metrics.auc(metric_xaxis=metrics.recall,
                          metric_yaxis=metrics.precision,
                          method=AggregationMethod.ONE_VS_ALL)
_auc_micro = stream_metrics.auc(metric_xaxis=metrics.recall,
                          metric_yaxis=metrics.precision,
                          method=AggregationMethod.MICRO)
_auc_macro = stream_metrics.auc(metric_xaxis=metrics.recall,
                          metric_yaxis=metrics.precision,
                          method=AggregationMethod.MACRO)

print("One VS All AUC of Precision Recall:")
print(_auc_onevsall)
print("Micro Averaged AUC of Precision Recall:")
print(_auc_micro)
print("Macro Averaged AUC of Precision Recall:")
print(_auc_macro)


# get metrics such as F1 at all thresholds
f1_scores = stream_metrics.calc_metric(metric=metrics.f1_score)
plt.plot(stream_metrics.thresholds, f1_scores)
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.show()


# ##### PERFORMANCE CURVES
# # plot one vs all precision recall curve for all classes
fig = stream_metrics.plot_precision_recall_curve(class_names=iris.target_names,
                                                 method=AggregationMethod.ONE_VS_ALL)
fig.suptitle("ONE_VS_ALL PR")
plt.show()
# plot one vs all precision recall curve for all classes
fig = stream_metrics.plot_roc_curve(class_names=iris.target_names,
                              method=AggregationMethod.ONE_VS_ALL)
fig.suptitle("ONE_VS_ALL ROC")
plt.show()

# plot one vs all precision recall curve for specific class
fig = stream_metrics.plot_precision_recall_curve(class_names=iris.target_names,
                              method=AggregationMethod.ONE_VS_ALL,
                                           class_index=0)
fig.suptitle("ONE_VS_ALL but only for CLASS 0")
plt.show()


# plot micro averaged precision recall
fig = stream_metrics.plot_precision_recall_curve(class_names=iris.target_names,
                              method=AggregationMethod.MICRO)
fig.suptitle("MICRO AVERAGED PR CURVE")
plt.show()

# plot Macro averaged precision recall
fig = stream_metrics.plot_precision_recall_curve(class_names=iris.target_names,
                              method=AggregationMethod.MACRO)
fig.suptitle("MACRO AVERAGED PR CURVE")
plt.show()

