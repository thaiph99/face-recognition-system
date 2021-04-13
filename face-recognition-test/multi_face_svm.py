# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""

import face_recognition
from sklearn import svm
from os import getcwd
import os
import json
import numpy as np
from json import JSONEncoder
import matplotlib.pyplot as plt

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
print(getcwd())
train_dir = os.listdir(getcwd() + '/train_dir')
print(train_dir)

# Loop through each person in the training directory

print('encoding face image')

for person in train_dir:
    pix = os.listdir(getcwd() + "/train_dir/" + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file(
            getcwd() + "/train_dir/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        # If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img +
                  " was skipped and can't be used for training")

# print(encodings[0])
# create dict to save to json
json_file = {}
for name in names:
    json_file[name] = []
for i in range(len(encodings)):
    json_file[names[i]].append(encodings[i])


# print(json_file)
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# print(json_file)

with open('face_data.json', 'w') as f:
    json.dump(json_file, f, cls=NumpyArrayEncoder)

with open('face_data.json', 'r') as f:
    data = json.load(f)
print(type(data))

vec_face = []
list_name = []
for name in data.keys():
    for fa in data[name]:
        vec_face.append(fa)
        list_name.append(name)

print(list_name)
print(len(vec_face))

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
print('training face')
clf.fit(vec_face, list_name)
print('done')

# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('img2.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

# Predict all the faces in the test image using the trained classifier
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    print(*name)
