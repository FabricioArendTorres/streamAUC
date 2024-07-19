import numpy as np
import unittest
from streamauc import StreamingMetrics, AggregationMethod, auc
from streamauc.metrics import f1_score, tpr, fpr
import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import (
    precision_recall_curve as sk_precision_recall_curve,
    roc_curve as sk_roc_curve,
    confusion_matrix,
)

_conf_matr_multiclass = np.array(
    [
        [[19, 10, 17], [0, 14, 17], [0, 3, 2]],
        [[5, 10, 10], [16, 2, 12], [6, 9, 12]],
        [[4, 0, 16], [3, 10, 8], [18, 13, 10]],
        [[0, 15, 19], [3, 20, 0], [3, 18, 4]],
    ]
)


class TestInit(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)

    def test_unsorted_thresholds(self):
        thresholds = np.linspace(0, 1, 100)
        np.random.shuffle(thresholds)

        self.assertFalse(np.all(thresholds[:-1] >= thresholds[1:]))
        curve = StreamingMetrics(
            thresholds=thresholds,
            num_classes=3,
        )
        self.assertTrue(np.all(curve.thresholds[:-1] >= curve.thresholds[1:]))

    def test_invalid_input(self):
        thresholds = np.arange(20)
        with self.assertRaises(ValueError):
            curve = StreamingMetrics(
                thresholds=thresholds,
                num_classes=3,
            )
        thresholds = thresholds[:1]
        with self.assertRaises(ValueError):
            curve = StreamingMetrics(
                thresholds=thresholds,
                num_classes=3,
            )
        with self.assertRaises(ValueError):
            curve = StreamingMetrics(
                num_thresholds=1,
                num_classes=3,
            )
        with self.assertRaises(ValueError):
            curve = StreamingMetrics(
                num_thresholds=10,
                num_classes=1,
            )


class TestConfusionMatrixUpdates(unittest.TestCase):
    def setUp(self):
        np.random.seed(1234)

    def test_reset(self):
        thresholds = np.linspace(0, 1, 100)
        curve = StreamingMetrics(
            thresholds=thresholds,
            num_classes=3,
        )
        expected_empty_confm = np.zeros(
            (len(thresholds) + 2, curve.num_classes, 2, 2),
            dtype=int,
        )
        np.testing.assert_array_equal(
            curve.confusion_matrix, expected_empty_confm
        )

        y_true = np.random.randint(0, 2, (10,))
        y_true = np.eye(curve.num_classes)[y_true]
        y_pred = np.random.random((10, curve.num_classes))
        y_pred = y_pred / y_pred.sum(-1, keepdims=True)

        curve.update(y_true=y_true, y_score=y_pred)
        conf1 = curve.confusion_matrix.copy()

        curve.reset()
        y_true = np.argmax(y_true, -1)
        curve.update(y_true=y_true, y_score=y_pred)
        conf2 = curve.confusion_matrix.copy()
        np.testing.assert_array_equal(conf1, conf2)

        self.assertFalse(
            np.allclose(curve.confusion_matrix, expected_empty_confm)
        )

        curve.reset()
        self.assertTrue(
            np.allclose(curve.confusion_matrix, expected_empty_confm)
        )

    def test_invalid_input(self):

        thresholds = np.linspace(0, 1, 100)
        curve = StreamingMetrics(
            thresholds=thresholds,
            num_classes=3,
        )
        with self.assertRaises(ValueError):
            y_true = np.random.randint(0, 2, (10, curve.num_classes))
            y_pred = np.random.randint(0, 2, (10, curve.num_classes + 1))
            curve.update(y_true=y_true, y_score=y_pred)

        with self.assertRaises(ValueError):
            y_true = np.random.randint(0, 2, (11, curve.num_classes))
            y_pred = np.random.randint(0, 2, (10, curve.num_classes))
            curve.update(y_true=y_true, y_score=y_pred)

        with self.assertRaises(ValueError):
            y_true = np.random.randint(0, 2, (10, curve.num_classes, 2, 2))
            y_pred = np.random.randint(0, 2, (10, curve.num_classes, 2, 2))
            curve.update(y_true=y_true, y_score=y_pred)
        with self.assertRaises(ValueError):
            y_true = np.random.randint(0, 2, (10, curve.num_classes))
            y_pred = np.random.randint(0, 2, (10, curve.num_classes, 2, 2))
            curve.update(y_true=y_true, y_score=y_pred)

        # should not throw any errors
        y_true = np.random.randint(0, 2, (10,))
        y_true = np.eye(curve.num_classes)[y_true][..., np.newaxis, np.newaxis]
        y_pred = np.random.randint(0, 2, (10, curve.num_classes))
        curve.update(y_true=y_true, y_score=y_pred)

        # should not throw any errors
        y_true = np.random.randint(0, 2, (10,))
        y_pred = np.random.randint(0, 2, (10, curve.num_classes))
        curve.update(y_true=y_true, y_score=y_pred)

        # should not throw any errors
        y_true = np.random.randint(0, 2, (10, 1, 1, 1, 1))
        y_pred = np.random.randint(0, 2, (10, curve.num_classes))
        curve.update(y_true=y_true, y_score=y_pred)

        # should not throw any errors
        y_true = np.random.randint(0, 2, (10, 1, 1, 1, 1))
        y_pred = np.random.randint(0, 2, (10, curve.num_classes))
        curve.update(y_true=y_true, y_score=y_pred, check_inputs=False)

        # y_true = np.random.randint(0, 2, (10, curve.num_classes, 1, 2))
        # y_pred = np.random.randint(0, 2, (10, curve.num_classes, 1, 2))
        # curve.update(y_true=y_true, y_pred=y_pred, check_inputs=False)


class TestMetricsMulticlass(unittest.TestCase):
    def setUp(self):
        self.num_thresholds = 3
        self.dim = 2
        self.confusion_matrix = _conf_matr_multiclass

        self.curve = StreamingMetrics(
            num_thresholds=self.num_thresholds,
            num_classes=self.dim,
        )
        self.curve._confusion_matrix = self.confusion_matrix

    def test_true_negatives(self):
        calculated_val = self.curve.calc_metric(
            metric=f1_score, method=AggregationMethod.MACRO
        )
        expected_val = f1_score(
            tp=self.curve.true_positives(),
            fn=self.curve.false_negatives(),
            fp=self.curve.false_positives(),
        )
        np.testing.assert_array_equal(expected_val, calculated_val)


class TestAUCMulticlass(unittest.TestCase):
    def setUp(self):

        iris = load_iris()
        X, y = iris.data, iris.target
        y = iris.target_names[y]

        random_state = np.random.RandomState(0)
        n_samples, n_features = X.shape
        X = np.concatenate(
            [X, random_state.randn(n_samples, 200 * n_features)], axis=1
        )
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

        classifier = LogisticRegression()
        self.y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

        label_binarizer = LabelBinarizer().fit(y_train)

        self.y_test = np.argmax(label_binarizer.transform(y_test), -1)

        thresholds = np.unique(self.y_score)
        self.dim = 3
        self.curve = StreamingMetrics(
            thresholds=thresholds,
            num_classes=self.dim,
        )

    def test_sklearn(self):

        for class_idx in range(self.dim):

            fpr_sk, tpr_sk, thresholds = metrics.roc_curve(
                self.y_test == class_idx, self.y_score[..., class_idx]
            )

            self.curve = StreamingMetrics(
                thresholds=thresholds[1:],
                num_classes=self.dim,
            )
            self.curve.reset()
            self.curve.update(y_true=self.y_test, y_score=self.y_score)
            #
            tpr_ours = self.curve.calc_metric(
                metric=tpr, method=AggregationMethod.ONE_VS_ALL
            ).squeeze()[:, class_idx]
            fpr_ours = self.curve.calc_metric(
                metric=fpr, method=AggregationMethod.ONE_VS_ALL
            ).squeeze()[:, class_idx]

            np.testing.assert_allclose(tpr_ours[:-1], tpr_sk)
            np.testing.assert_allclose(fpr_ours[:-1], fpr_sk)
            self.assertEqual(tpr_sk.shape[0], tpr_ours.shape[0] - 1)

            auc_sk = metrics.auc(fpr_sk, tpr_sk)
            auc_custom = auc(fpr_ours, tpr_ours)
            auc_curve = self.curve.auc(class_index=class_idx)

            self.assertTrue(np.isclose(auc_custom, auc_sk))
            self.assertTrue(np.isclose(auc_sk, auc_curve))

        self.curve.false_positives()


class TestStreamingMetrics(unittest.TestCase):
    def setUp(self):
        cancer_ds = load_breast_cancer()
        X, y = cancer_ds.data, cancer_ds.target

        random_state = np.random.RandomState(0)
        n_samples, n_features = X.shape
        X = np.concatenate(
            [X, random_state.randn(n_samples, 200 * n_features)], axis=1
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, stratify=y, random_state=0
        )

        classifier = LogisticRegression(max_iter=1000)
        self.y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

        self.y_test = y_test

        thresholds = np.unique(self.y_score)
        self.dim = 2
        self.curve = StreamingMetrics(
            thresholds=thresholds,
            num_classes=self.dim,
        )

        # check that multiple updates have the same effect as one big..
        half = self.y_test.shape[0] // 2
        self.curve.update(self.y_test[:half], self.y_score[:half])
        self.curve.update(self.y_test[half:], self.y_score[half:])

    def test_total(self):
        new_curve = StreamingMetrics(
            num_thresholds=100,
            num_classes=self.dim,
        )

        self.assertEqual(new_curve._total().shape, (100, self.dim))
        np.testing.assert_allclose(
            new_curve._total(), np.zeros_like(new_curve._total())
        )

        new_curve.update(self.y_test, self.y_score)
        new_curve.update(self.y_test, self.y_score)
        self.assertEqual(new_curve._total().shape, (100, self.dim))

        np.testing.assert_allclose(
            new_curve._total(),
            2 * self.y_test.shape[0] * np.ones_like(new_curve._total()),
        )

    def test_confusion_matrix(self):
        for class_idx in range(self.dim):
            y_true = self.y_test == class_idx

            for threshold in self.curve.thresholds:
                y_pred = self.y_score[:, class_idx] >= threshold

                # sklearn has the confusion matrix flipped
                confm_ref = np.flip(confusion_matrix(y_true, y_pred))

                computed_confm = self.curve.confusion_matrix[
                    self.curve.thresholds.tolist().index(threshold), class_idx
                ]
                np.testing.assert_array_equal(computed_confm, confm_ref)

    def test_precision_recall_curve(self):
        for class_idx in range(self.dim):
            precision, recall, thresholds = sk_precision_recall_curve(
                self.y_test == class_idx, self.y_score[:, class_idx]
            )

            new_curve = StreamingMetrics(
                thresholds=thresholds,
                num_classes=self.dim,
            )

            # check that multiple updates have the same effect as one big..
            half = self.y_test.shape[0] // 2
            new_curve.update(self.y_test[:half], self.y_score[:half])
            new_curve.update(self.y_test[half:], self.y_score[half:])
            stream_prec, stream_recall, stream_thresholds = (
                new_curve.precision_recall_curve(class_index=class_idx)
            )
            np.testing.assert_almost_equal(stream_thresholds[1:], thresholds)
            np.testing.assert_almost_equal(precision[:1], stream_prec[:1])
            np.testing.assert_almost_equal(recall, stream_recall)

    def test_roc_curve(self):
        for class_idx in range(self.dim):
            _fpr, _tpr, thresholds = sk_roc_curve(
                self.y_test == class_idx, self.y_score[:, class_idx]
            )

            new_curve = StreamingMetrics(
                thresholds=thresholds[1:],
                num_classes=self.dim,
            )

            # ensure that multiple updates have the same effect as one big..
            half = self.y_test.shape[0] // 2
            new_curve.update(self.y_test[:half], self.y_score[:half])
            new_curve.update(self.y_test[half:], self.y_score[half:])

            streaming_fpr, streaming_tpr, _thr = new_curve.roc_curve(
                class_index=class_idx
            )
            np.testing.assert_almost_equal(_fpr, streaming_fpr[:-1])
            np.testing.assert_almost_equal(_tpr, streaming_tpr[:-1])


class TestStreamingMetricsBinary(unittest.TestCase):
    def setUp(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        y = iris.target_names[y]

        random_state = np.random.RandomState(0)
        n_samples, n_features = X.shape
        X = np.concatenate(
            [X, random_state.randn(n_samples, 200 * n_features)], axis=1
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, stratify=y, random_state=0
        )

        classifier = LogisticRegression(max_iter=1000)
        self.y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

        label_binarizer = LabelBinarizer().fit(y_train)
        self.y_test = np.argmax(label_binarizer.transform(y_test), -1)

        thresholds = np.unique(self.y_score)
        self.dim = 3
        self.curve = StreamingMetrics(
            thresholds=thresholds,
            num_classes=self.dim,
        )

        # check that multiple updates have the same effect as one big..
        half = self.y_test.shape[0] // 2
        self.curve.update(self.y_test[:half], self.y_score[:half])
        self.curve.update(self.y_test[half:], self.y_score[half:])

    def test_total(self):
        new_curve = StreamingMetrics(
            num_thresholds=100,
            num_classes=self.dim,
        )

        self.assertEqual(new_curve._total().shape, (100, self.dim))
        np.testing.assert_allclose(
            new_curve._total(), np.zeros_like(new_curve._total())
        )

        new_curve.update(self.y_test, self.y_score)
        new_curve.update(self.y_test, self.y_score)
        self.assertEqual(new_curve._total().shape, (100, self.dim))

        np.testing.assert_allclose(
            new_curve._total(),
            2 * self.y_test.shape[0] * np.ones_like(new_curve._total()),
        )

    def test_confusion_matrix(self):
        for class_idx in range(self.dim):
            y_true = self.y_test == class_idx

            for threshold in self.curve.thresholds:
                y_pred = self.y_score[:, class_idx] >= threshold

                # sklearn has the confusion matrix flipped
                confm_ref = np.flip(confusion_matrix(y_true, y_pred))

                computed_confm = self.curve.confusion_matrix[
                    self.curve.thresholds.tolist().index(threshold), class_idx
                ]
                np.testing.assert_array_equal(computed_confm, confm_ref)

    def test_precision_recall_curve(self):
        for class_idx in range(self.dim):
            precision, recall, thresholds = sk_precision_recall_curve(
                self.y_test == class_idx, self.y_score[:, class_idx]
            )

            new_curve = StreamingMetrics(
                thresholds=thresholds,
                num_classes=self.dim,
            )

            # check that multiple updates have the same effect as one big..
            half = self.y_test.shape[0] // 2
            new_curve.update(self.y_test[:half], self.y_score[:half])
            new_curve.update(self.y_test[half:], self.y_score[half:])
            stream_prec, stream_recall, stream_thresholds = (
                new_curve.precision_recall_curve(class_index=class_idx)
            )
            np.testing.assert_almost_equal(stream_thresholds[1:], thresholds)
            np.testing.assert_almost_equal(precision[:1], stream_prec[:1])
            np.testing.assert_almost_equal(recall, stream_recall)

    def test_roc_curve(self):
        for class_idx in range(self.dim):
            _fpr, _tpr, thresholds = sk_roc_curve(
                self.y_test == class_idx, self.y_score[:, class_idx]
            )

            new_curve = StreamingMetrics(
                thresholds=thresholds[1:],
                num_classes=self.dim,
            )

            # ensure that multiple updates have the same effect as one big..
            half = self.y_test.shape[0] // 2
            new_curve.update(self.y_test[:half], self.y_score[:half])
            new_curve.update(self.y_test[half:], self.y_score[half:])

            streaming_fpr, streaming_tpr, _thr = new_curve.roc_curve(
                class_index=class_idx
            )
            np.testing.assert_almost_equal(_fpr, streaming_fpr[:-1])
            np.testing.assert_almost_equal(_tpr, streaming_tpr[:-1])


class TestStreamingIdentities(unittest.TestCase):
    def setUp(self):
        iris = load_iris()
        X, y = iris.data, iris.target
        y = iris.target_names[y]

        random_state = np.random.RandomState(0)
        n_samples, n_features = X.shape
        X = np.concatenate(
            [X, random_state.randn(n_samples, 200 * n_features)], axis=1
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, stratify=y, random_state=0
        )

        classifier = LogisticRegression(max_iter=1000)
        self.y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

        label_binarizer = LabelBinarizer().fit(y_train)
        self.y_test = np.argmax(label_binarizer.transform(y_test), -1)

        thresholds = np.unique(self.y_score)
        self.dim = 3
        self.curve = StreamingMetrics(
            thresholds=thresholds,
            num_classes=self.dim,
        )

        # check that multiple updates have the same effect as one big..
        half = self.y_test.shape[0] // 2
        self.curve.update(self.y_test[:half], self.y_score[:half])
        self.curve.update(self.y_test[half:], self.y_score[half:])

    def test_total(self):
        total = self.curve._total().squeeze()

        tp = self.curve.true_positives()
        fp = self.curve.false_positives()
        tn = self.curve.true_negatives()
        fn = self.curve.false_negatives()

        pp = self.curve.predicted_positives()
        pn = self.curve.predicted_negatives()
        p = self.curve.positives()
        n = self.curve.negatives()

        np.testing.assert_array_equal(total, tp + fp + tn + fn)
        np.testing.assert_array_equal(p, tp + fn)
        np.testing.assert_array_equal(n, tn + fp)
        np.testing.assert_array_equal(pp, tp + fp)
        np.testing.assert_array_equal(pn, tn + fn)


if __name__ == "__main__":
    unittest.main()
