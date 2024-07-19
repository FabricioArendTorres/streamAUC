# streamauc

[![codecov](https://codecov.io/gh/FabricioArendTorres/streamAUC/branch/main/graph/badge.svg?token=streamAUC_token_here)](https://codecov.io/gh/FabricioArendTorres/streamAUC)
[![CI](https://github.com/FabricioArendTorres/streamAUC/actions/workflows/main.yml/badge.svg)](https://github.com/FabricioArendTorres/streamAUC/actions/workflows/main.yml)

## Multi-Class Classification Metrics from data streams and minibatches

A low dependency python package for keeping track of classification metrics 
such as AUC given probabilistic outputs.

In essence, the package keeps track of one-vs-all confusion matrices for each 
class for a range of thresholds. 
This allows a minibatch based updating of the things such as ROC or 
Precision-Recall curves, without having to store all the predictions.
Metrics can then be computed either in a one-vs-all fashion, or by micro- 
or macro averaging.

My main usage is for multiclass semantic segmentation, where the train and 
test data becomes rather large for pixel-wise performance metrics.

This package supports a range of classical performance metrics, such as:
- TPR, FNR, FPR, TNR, Accuracy, F1-Score, Jaccard Index, ...
- Corresponding curves, such as Precision-Recall (PR) curves or ROC curves.
- AUC of ROC and PR curves, or any combination of two metrics you want.
- One-vs-all, micro, or macro averaging of metrics for a set of predefined 
  thresholds.

## Lightweight, tested, and permissive License

- Only Numpy and Matplotlib are requirements.
- High Test Coverage: Metrics are unit tested against sklearn metrics.
- Permissive License: Licensed under Apache 2.0.

## Installation

### Pypi Current Release
```bash
pip install streamauc
```

### Latest Version from Github
```bash
pip install git+https://github.com/FabricioArendTorres/streamAUC.git
```


## Usage
Below you can find pseudocode for the usage of this package.
For a more comprehensive and self-consistent example, see `examples/example.py`.

### Keep track of confusion matrices at many thresolds

```py
import numpy as np

from streamauc import StreamingMetrics, AggregationMethod

# Select the number of thresholds for which we want to keep track of results.
stream_metrics = StreamingMetrics(
  thresholds=np.linspace(0, 1, 200),
  num_classes=10,
)

while youhavedata:
  y_true = ...  # true classes, shape (-1,) or one-hot-encoded (-1,num_classes)
  pred_prob_y = ...  # indicating class probabilities,  shape (-1, num_classes), 
  stream_metrics.update(y_true=y_true, y_score=pred_prob_y)

## get 1-vs-all confusion matrix at all thresholds
confm = stream_metrics.confusion_matrix 
# confm is of shape (num_thresholds, num_classes, 2, 2)

## get metrics at all thresholds
tp = stream_metrics.true_positives()  # is of shape (num_threholds, num_classes)

fpr, tpr, thresholds = stream_metrics.roc_curve(
  AggregationMethod.ONE_VS_ALL) # fpr and tpr are of shape (num_thresholds, num_classes)


fpr, tpr, thresholds = stream_metrics.precision_recall_curve(
  AggregationMethod.MACRO) # fpr and tpr are of shape (num_thresholds, )

# reset before updating with new data
stream_metrics.reset()

```

### Track metrics in a minibatch based training loop
```py
import matplotlib.pyplot as plt

from streamauc import StreamingMetrics, AggregationMethod, auc
from streamauc import metrics

# Select the number of thresholds for which we want to keep track of results.
stream_metrics = StreamingMetrics(
  num_thresholds=100,
  num_classes=3,
)

# Whatever your model may be, you need probabilities for the 
# defined number of classes.
model = ...
yourdataiterator = ...

for epoch in range(100):
  ... # do your training step
  
  for mb_x, mb_y in yourdataiterator:
    pred_prob_y = model.predict_proba(mb_x) # of shape (-1, num_classes)
    # mb_y can be onehot encoded (-1, num_classes) or a flat integer array (-1,)
    stream_metrics.update(y_true=mb_y, y_score=pred_prob_y)
  
  # compute metrics you want
  _auc_macro = stream_metrics.auc(metrics.recall,
                                  metrics.precision,
                                  method=AggregationMethod.MACRO)
  f1_for_all_thresholds = stream_metrics.calc_metric(metric=metrics.f1_score)
  
  # Plot all 1-vs-all/micro-averaged/macro-averaged Precision Recall Curves
  fig = stream_metrics.precision_recall_curve(method=AggregationMethod.
                                              ONE_VS_ALL)
  fig.savefig(f"PR_one_vs_all_{epoch}.png")
  plt.close(fig)
  
  # reset the tracker for the next epoch 
  stream_metrics.reset()
```




## Things to note

### Curves and AUC are only approximate
StreamAUC works by keeping track of confusion matrices at different 
thresholds, which are defined at the beginning. That is, the resulting 
curves and AUC are by construction always approximations.

This should however not be too limiting for any application with large data 
sets, as in that case the number of unique thresholds becomes infeasible in 
any case.

### Precision-Recall Curve: Definition of precision when recall is zero
There are different conventions regarding the precision when there are no 
positive predictions, which occurs at the left-most point of the 
precision-recall curve corresponding to a threshold of 1. 
Technically, its undefined, since we have TP/(TP+FP)=0/0. 
Scikit-learn then defines it as 1, for the sake of nicer PR curves.
This package defines it as 0, as a value of 1 seems misleading in my opinion.

## Custom Metrics
It's straight-forward to add custom metrics to this package, just define a 
function with the following interface, which can then be passed as Callable to 
`StreamingMetrics.calc_metric`,  `StreamingMetrics.auc`.
The basic metrics (TP, FN, FP, TN) are always in the shape of `
(num_thresholds, num_classes)`, with e.g. `TP[:,2]` corresponding to the 
number of true positives at each threshold in a one-vs-all setting for the 
class with index 2.

See for example the F1 metric implementation for the required interface:
```python
from typing import Optional
import numpy as np

from streamauc.utils import AggregationMethod, check_confusion_matrix_entries

def custom_f1_score(
    tp: np.ndarray,
    fn: np.ndarray,
    fp: np.ndarray,
    tn: np.ndarray,
    method: AggregationMethod = AggregationMethod.MACRO,
    class_index: Optional[int] = None,
    check_inputs: bool = True,
):  
    
    if check_inputs:
        # do some optional checks for valid shapes etc.
        check_confusion_matrix_entries(tp, fn, fp, tn)
        
    if method == AggregationMethod.MICRO:
        tp_sum = np.sum(tp, axis=-1)
        fn_sum = np.sum(fn, axis=-1)
        fp_sum = np.sum(fp, axis=-1)
        _f1 = ((2 * tp_sum) / (2 * tp_sum + fp_sum + fn_sum + 1e-12))
        _f1 = _f1[..., class_index]
    elif method == AggregationMethod.MACRO:
        _f1 = ((2 * tp) / (2 * tp + fp + fn + 1e-12)).mean(-1)
    elif method == AggregationMethod.ONE_VS_ALL:
        _f1 = ((2 * tp) / (2 * tp + fp + fn + 1e-12))[..., class_index]
    else:
        raise ValueError(
            f"Method must one of {[e.value for e in AggregationMethod]}. "
            f"Got {method}."
        )
    return _f1
```



