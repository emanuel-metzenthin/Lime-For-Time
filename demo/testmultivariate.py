import logging
import numpy as np

from sklearn.pipeline import Pipeline

from sktime.classification.compose import TimeSeriesForestClassifier

from sktime.transformations.panel.compose import ColumnConcatenator

import matplotlib.pyplot as plt

from lime import explanation
from lime import lime_base
import lime_timeseries as lime_ts

# random dataset
def genDataset(num_samples, num_channels, num_points):
    X = np.random.random((num_samples, num_channels, num_points))
    y = np.array([np.random.randint(0,2) for _ in range(num_samples)])
    signal_names = ["chan_%d" % x for x in range(num_channels)]
    return signal_names, X, y

def testlime(signal_names, clf, x, y):
    class_names=[y]

    num_slices=20
    num_features=10

    explainer = lime_ts.LimeTimeSeriesExplainer(class_names=class_names,
                                                signal_names=signal_names)

    labelid = 0
    exp = explainer.explain_instance(x, clf.predict_proba, num_features=num_features, num_samples=100, num_slices=num_slices, labels=[labelid], replacement_method='total_mean')
    exp.as_pyplot_figure(labelid)
    plt.show()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    
    signal_names, X, y = genDataset(80, 4, 30)

    steps = [
        ("concatenate", ColumnConcatenator()),
        ("classify", TimeSeriesForestClassifier(n_estimators=100)),
    ]
    clf = Pipeline(steps)
    clf.fit(X,y)

    testlime(signal_names, clf, X[0], y[0])
