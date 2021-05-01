from api import Model
import numpy as np
import os
import face_recognition
from sklearn import svm
import json
from json import JSONEncoder
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


path_train = '/home/thai/Data2/face-recognition-system/face-recognition-test/processed_data/train1'
path_validation = '/home/thai/Data2/face-recognition-system/face-recognition-test/processed_data/validation'


# read data
def read_data(path):
    names = []
    imgs = []
    pix = os.listdir(path)
    for name in pix:
        path1 = path + '/' + name
        pix1 = os.listdir(path1)
        for img in pix1:
            path2 = path1 + '/' + img
            face = face_recognition.load_image_file(path2)
            face_bounding_boxes = face_recognition.face_locations(face)
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                imgs.append(face_enc)
                names.append(name)
        print(name)
    return imgs, names


print('reading data')
train_X, train_y = read_data(path_train)
valid_X, valid_y = read_data(path_validation)

filename = '/home/thai/Data2/face-recognition-system/storage/face_data.json'
with open(filename, 'r') as f:
    data = json.load(f)
faces_encoded = []
faces_name = []
for name in data.keys():
    for fa in data[name]:
        faces_encoded.append(fa)
        faces_name.append(name)

faces_encoded += train_X
faces_name += train_y

print(set(faces_name))
json_file = {}
for name in faces_name:
    json_file[name] = []
for i in range(len(faces_encoded)):
    json_file[faces_name[i]].append(faces_encoded[i])

filename = filename
with open(filename, 'w') as f:
    json.dump(json_file, f, cls=NumpyArrayEncoder)

print('training data')
clf = svm.SVC(gamma='scale')
clf.fit(train_X, train_y)

print('predicting')
y_pred = clf.predict(valid_X)

print('validating')
print('Model accuracy score with sigmoid kernel and C=1.0 : {0:0.4f}'.format(accuracy_score(valid_y, y_pred)))
