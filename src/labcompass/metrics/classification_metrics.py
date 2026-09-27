import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_accuracy(pred, target):
    """Compute the classification accuracy between softmax-derived predicted labels and `target`.

    `pred` is converted to class probabilities with a softmax over the last axis, and the predicted
    label for each sample is taken as the `argmax` of those probabilities.

    :param pred: Predicted logits/scores, of shape `(num_samples, num_classes)`.
    :type pred: class:`TensorLike`

    :param target: Ground-truth class labels, of shape `(num_samples,)`.
    :type target: class:`TensorLike`

    :return: The fraction of samples for which the predicted label matches `target`, as computed by
        :func:`sklearn.metrics.accuracy_score`.
    :rtype: class:`float`
    """
    probs = softmax(pred, axis=-1)
    label = probs.argmax(-1)
    return accuracy_score(target, label)


def compute_precision(pred, target, average="weighted"):
    """Compute the precision score between softmax-derived predicted labels and `target`.

    `pred` is converted to class probabilities with a softmax over the last axis, and the predicted
    label for each sample is taken as the `argmax` of those probabilities.

    :param pred: Predicted logits/scores, of shape `(num_samples, num_classes)`.
    :type pred: class:`TensorLike`

    :param target: Ground-truth class labels, of shape `(num_samples,)`.
    :type target: class:`TensorLike`

    :param average: Averaging strategy forwarded to :func:`sklearn.metrics.precision_score`
        (e.g. `"micro"`, `"macro"`, `"weighted"`), defaults to `"weighted"`.
    :type average: class:`str`

    :return: The precision score, as computed by :func:`sklearn.metrics.precision_score`.
    :rtype: class:`float`
    """
    probs = softmax(pred, axis=-1)
    label = probs.argmax(-1)
    return precision_score(target, label, average=average)


def compute_recall(pred, target, average="weighted"):
    """Compute the recall score between softmax-derived predicted labels and `target`.

    `pred` is converted to class probabilities with a softmax over the last axis, and the predicted
    label for each sample is taken as the `argmax` of those probabilities.

    :param pred: Predicted logits/scores, of shape `(num_samples, num_classes)`.
    :type pred: class:`TensorLike`

    :param target: Ground-truth class labels, of shape `(num_samples,)`.
    :type target: class:`TensorLike`

    :param average: Averaging strategy forwarded to :func:`sklearn.metrics.recall_score`
        (e.g. `"micro"`, `"macro"`, `"weighted"`), defaults to `"weighted"`.
    :type average: class:`str`

    :return: The recall score, as computed by :func:`sklearn.metrics.recall_score`.
    :rtype: class:`float`
    """
    probs = softmax(pred, axis=-1)
    label = probs.argmax(-1)
    return recall_score(target, label, average=average)


def compute_f1(pred, target, average="weighted"):
    """Compute the F1 score between softmax-derived predicted labels and `target`.

    `pred` is converted to class probabilities with a softmax over the last axis, and the predicted
    label for each sample is taken as the `argmax` of those probabilities.

    :param pred: Predicted logits/scores, of shape `(num_samples, num_classes)`.
    :type pred: class:`TensorLike`

    :param target: Ground-truth class labels, of shape `(num_samples,)`.
    :type target: class:`TensorLike`

    :param average: Averaging strategy forwarded to :func:`sklearn.metrics.f1_score`
        (e.g. `"micro"`, `"macro"`, `"weighted"`), defaults to `"weighted"`.
    :type average: class:`str`

    :return: The F1 score, as computed by :func:`sklearn.metrics.f1_score`.
    :rtype: class:`float`
    """
    probs = softmax(pred, axis=-1)
    label = probs.argmax(-1)
    return f1_score(target, label, average=average)


def compute_roc_auc(pred, target, average="micro"):
    """Compute the multi-class ROC-AUC score between softmax-derived predicted probabilities and `target`.

    The unique classes found in `target` are used to build a one-hot encoding of the ground-truth labels
    (via :func:`sklearn.preprocessing.label_binarize`), while `pred` is converted to class probabilities
    with a softmax over the last axis.

    :param pred: Predicted logits/scores, of shape `(num_samples, num_classes)`.
    :type pred: class:`TensorLike`

    :param target: Ground-truth class labels, of shape `(num_samples,)`.
    :type target: class:`TensorLike`

    :param average: Averaging strategy forwarded to :func:`sklearn.metrics.roc_auc_score`
        (e.g. `"micro"`, `"macro"`), defaults to `"micro"`.
    :type average: class:`str`

    :return: The area under the ROC curve, as computed by :func:`sklearn.metrics.roc_auc_score`.
    :rtype: class:`float`
    """
    classes = np.unique(target)
    y_true_bin = label_binarize(
        [np.where(classes == c)[0][0] for c in target], classes=range(len(classes))
    )
    probs = softmax(pred, axis=-1)
    return roc_auc_score(y_true_bin, probs, average=average)
