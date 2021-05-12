# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from numpy.linalg import norm
import numpy as np
from math import e, log


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


# load faces
data = load('face.npz')
testX_faces = data['arr_2']
# load face embeddings
data = load('face_embeded.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# create dict ( key == name , value = embedded)
# print(trainy)
dict_face = {}
for i in range(len(trainX)):
    if trainy[i] not in dict_face.keys():
        dict_face[trainy[i]] = []
    dict_face[trainy[i]].append(trainX[i])

# print(dict_face)


def cal_similarity_score(embed, name):
    embed = np.array(embed)
    list_norm = []
    for em in dict_face[name]:
        print('em :', em.shape, sum(em))
        print('embed:', embed.shape, sum(embed))
        list_norm.append(norm(em-embed, ord=2))
    print('list norm', list_norm)
    return sum(list_norm)/len(list_norm)


# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)
# test model on a random example from the test dataset
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0, class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print(len(samples))
print('entropy', entropy(yhat_prob[0]))
print('yhat probabiliti', yhat_prob)

print('similarity score', cal_similarity_score(
    samples[0], random_face_name[0]))
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])


print('test norm : ')
for emb in trainX:
    print(norm(random_face_emb-emb, ord=2))


# plot for fun
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()
