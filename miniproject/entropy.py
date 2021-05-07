import face_recognition
from sklearn import svm
import os
import json
import numpy as np
from numpy import asarray, expand_dims
from numpy.linalg import norm
from json import JSONEncoder
import pickle
from deepface import DeepFace
from deepface.commons import functions, realtime, distance as dst
from mtcnn import MTCNN
import cv2
from PIL import Image
import Facenet
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pickle
from math import log, e


def entropy(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)
    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent

print(entropy())
