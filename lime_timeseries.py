import numpy as np
import sklearn
from lime import explanation
from lime import lime_base
import math


class LimeTimeSeriesExplainer(object):
    """Explains time series classifiers."""

    def __init__(self,
                 kernel_width=25,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 ):
        """Init function.
        Args:
            kernel_width: kernel width for the exponential kernel
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
            classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
        """

        # exponential kernel
        def kernel(d): return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.base = lime_base.LimeBase(kernel, verbose)
        self.class_names = class_names
        self.feature_selection = feature_selection

    def explain_instance(self,
                         timeseries_instance,
                         classifier_fn,
                         num_slices,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         model_regressor=None,
                         replacement_method='mean'):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).
        As distance function DTW metric is used.

        Args:
            time_series_instance: time series to be explained.
            classifier_fn: classifier prediction probability function,
                which takes a list of d arrays with time series values
                and outputs a (d, k) numpy array with prediction
                probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            num_slices: Defines into how many slices the time series will
                be split up
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
            the K labels with highest prediction probabilities, where K is
            this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter to
                model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
       """

        domain_mapper = explanation.DomainMapper()
        permutations, predictions, distances = self.__data_labels_distances(
            timeseries_instance, classifier_fn,
            num_samples, num_slices, replacement_method)

        if self.class_names is None:
            self.class_names = [str(x) for x in range(predictions[0].shape[0])]

        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names)
        ret_exp.predict_proba = predictions[0]

        if top_labels:
            labels = np.argsort(predictions[0])[-top_labels:]
            ret_exp.top_labels = list(predictions)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[int(label)],
             ret_exp.local_exp[int(label)],
             ret_exp.score,
             ret_exp.local_pred) = self.base.explain_instance_with_data(
                permutations, predictions,
                distances, label,
                num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def __data_labels_distances(cls,
                                timeseries,
                                classifier_fn,
                                num_samples,
                                num_slices,
                                replacement_method='mean'):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing slices from the
        time series and replacing them with other data points (specified by
        replacement_method: mean over slice range, mean of entire series or
        random noise). Then predicts with the classifier.

        Args:
            timeseries: Time Series to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a time series and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear
                model (perturbation + original time series)
            num_slices: how many slices the time series will be split into
                for discretization.
            replacement_method:  Defines how individual slice will be
                deactivated (can be 'mean', 'total_mean', 'noise')
        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of slices in the time series. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: distance between the original instance and
                    each perturbed instance
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0].reshape([1, -1]), metric='cosine').ravel() * 100

        values_per_slice = math.ceil(len(timeseries) / num_slices)
        deact_per_slice = np.random.randint(1, num_slices + 1, num_samples - 1)
        perturbation_matrix = np.ones((num_samples, num_slices))
        features_range = range(num_slices)
        original_data = [timeseries.copy()]

        for i, num_inactive in enumerate(deact_per_slice, start=1):
            # choose random slices indexes to deactivate
            inactive_idxs = np.random.choice(features_range, num_inactive,
                                             replace=False)
            perturbation_matrix[i, inactive_idxs] = 0
            tmp_series = timeseries.copy()

            for idx in inactive_idxs:
                start_idx = idx * values_per_slice
                end_idx = start_idx + values_per_slice
                end_idx = min(end_idx, len(timeseries))

                if replacement_method == 'mean':
                    # use mean of slice as inactive
                    tmp_series[start_idx:end_idx] = np.mean(
                        tmp_series[start_idx:end_idx])
                elif replacement_method == 'noise':
                    # use random noise as inactive
                    tmp_series[start_idx:end_idx] = np.random.uniform(
                        tmp_series.min(),
                        tmp_series.max(),
                        end_idx - start_idx)
                elif replacement_method == 'total_mean':
                    # use total series mean as inactive
                    tmp_series[start_idx:end_idx] = tmp_series.mean()
            original_data.append(tmp_series)

        predictions = classifier_fn(original_data)
        distances = distance_fn(perturbation_matrix)

        return perturbation_matrix, predictions, distances
