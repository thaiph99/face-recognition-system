import face_recognition
from sklearn import svm
import os
import json
import numpy as np
from json import JSONEncoder
import pickle


class Model:
    def __init__(self, app):
        # for training
        self.storage_temporary = app.config['storage_temporary']
        self.storage = app.config["storage"]
        # for recognition
        self.faces_encoded = []  # faces data for recognition
        self.faces_name = []
        self.model = None

        self.__load_model()
        self.__load_data()

    def __load_model(self):
        filename = self.storage + 'svm_model'
        self.model = pickle.load(open(filename, 'rb'))

    def __load_data(self):
        filename = self.storage + 'faces_data.json'
        with open(filename, 'r') as f:
            data = json.load(f)

        for name in data.keys():
            for fa in data[name]:
                self.faces_encoded.append(fa)
                self.faces_name.append(name)

    def __save_date(self):
        filename = self.storage + 'faces_data.json'
        with open('face_data.json', 'w') as f:
            json.dump(filename, f, cls=NumpyArrayEncoder)

    def __append_new_face(self, name):
        filename = self.storage_temporary
        pix = os.listdir(filename)

        for img in pix:
            face = face_recognition.load_image_file(filename + '/' + img)
            face_bounding_boxes = face_recognition.face_locations(face)
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                self.faces_encoded.append(face_enc)
                self.faces_name.append(name)

    def save_model(self):
        filename = self.storage + 'svm_model'
        pickle.dump(self.model, open(filename, 'wb'))

    def train(self, name):
        """
        :param name:
        :return: 1 if train success else 0
        """
        if name in self.faces_name:
            return 0

        self.__append_new_face(name)
        self.model = svm.SVC(gamma='scale')
        self.model.fit(self.faces_encoded, self.faces_name)
        self.save_model()
        return 1

    def recognize(self):
        """
        :return: 1 if detect old faces else 0
        """
        

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
