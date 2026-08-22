import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, roc_curve, auc
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
import torch


def compute_accuracy(pred, target):
    probs = softmax(pred, axis=-1)
    label = probs.argmax(-1)
    return accuracy_score(target, label)


def compute_precision(pred, target, average="weighthed"):
    probs = softmax(pred, axis=-1)
    label = probs.argmax(-1)
    return precision_score(target, label, average=average)


def compute_recall(pred, target, average="weighthed"):
    probs = softmax(pred, axis=-1)
    label = probs.argmax(-1)
    return recall_score(target, label, average=average)


def compute_f1(pred, target, average="weighthed"):
    probs = softmax(pred, axis=-1)
    label = probs.argmax(-1)
    return f1_score(target, label, average=average)


def compute_roc_auc(pred, target, average="micro"):
    classes = np.unique(target)
    y_true_bin = label_binarize(
        [np.where(classes == c)[0][0] for c in target], classes=range(len(classes))
    )
    probs = softmax(pred, axis=-1)
    return roc_auc_score(y_true_bin, probs, average=average)
