__author__ = 'thaiph99'

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
        self.faces_encoded = []
        self.faces_name = []
        # for recognition
        self.model = None

        self.__load_model()
        self.__load_data()

    def __load_model(self):
        filename = self.storage + '/' + 'svm_model.sav'
        self.model = pickle.load(open(filename, 'rb'))

    def __load_data(self):
        filename = self.storage + '/' + 'face_data.json'
        with open(filename, 'r') as f:
            data = json.load(f)
        self.faces_encoded = []
        self.faces_name = []
        for name in data.keys():
            for fa in data[name]:
                self.faces_encoded.append(fa)
                self.faces_name.append(name)

    def __save_date(self):
        filename = self.storage + 'faces_data.json'
        with open('face_data.json', 'w') as f:
            json.dump(filename, f, cls=NumpyArrayEncoder)

    def delete_face(self, name_del):
        json_file = {}
        for name in self.faces_name:
            if name == name_del:
                continue
            json_file[name] = []
        for i in range(len(self.faces_encoded)):
            if self.faces_name[i] == name_del:
                continue
            json_file[self.faces_name[i]].append(self.faces_encoded[i])

        filename = self.storage + '/' + 'face_data.json'
        with open(filename, 'w') as f:
            json.dump(json_file, f, cls=NumpyArrayEncoder)
        self.__load_model()
        self.__load_data()
        self.train_again()

    def __append_new_face(self, name):
        filename = self.storage_temporary
        pix = os.listdir(filename)

        for img in pix:
            if img == '.gitignore':
                continue
            face = face_recognition.load_image_file(filename + '/' + img)
            face_bounding_boxes = face_recognition.face_locations(face)
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                self.faces_encoded.append(face_enc)
                self.faces_name.append(name)
            # clear storage temporary
            os.remove(filename + '/' + img)

    def __save_model(self):
        filename = self.storage + '/' + 'svm_model.sav'
        pickle.dump(self.model, open(filename, 'wb'))

    def __save_data(self):
        json_file = {}
        for name in self.faces_name:
            json_file[name] = []
        for i in range(len(self.faces_encoded)):
            json_file[self.faces_name[i]].append(self.faces_encoded[i])

        filename = self.storage + '/' + 'face_data.json'
        with open(filename, 'w') as f:
            json.dump(json_file, f, cls=NumpyArrayEncoder)

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
        self.__save_model()
        self.__save_data()
        return 1

    def train_again(self):
        self.model = svm.SVC(gamma='scale')
        self.model.fit(self.faces_encoded, self.faces_name)
        self.__save_model()
        self.__save_data()

    def is_unknown_faces(self, unknown_face):
        """
        :param unknown_face:
        :return: True if unknown face else False
        """
        list_cmp = [self.faces_encoded[0]]
        print(self.faces_name[0], end=', ')
        for i in range(1, len(self.faces_name)):
            if self.faces_name[i - 1] != self.faces_name[i]:
                list_cmp.append(self.faces_encoded[i])
                print(self.faces_name[i], end=', ')
        print('\n')
        results = face_recognition.compare_faces(list_cmp, unknown_face)
        print(results)
        return True not in results

    def recognize(self):
        """
        :return: name of old people in image
        """
        filename = ''
        filename_dir = self.storage_temporary
        for i in os.listdir(filename_dir):
            if i != '.gitignore':
                filename = i
                break

        path = self.storage_temporary + '/' + filename
        test_img = face_recognition.load_image_file(path)
        test_img_encs = face_recognition.face_encodings(test_img)
        list_name_predict = []
        name = ''
        for face_enc in test_img_encs:
            if self.is_unknown_faces(face_enc):
                name = ['unknown face']
            else:
                name = self.model.predict(face_enc.reshape(1, -1))
            list_name_predict += list(name)
        print(list_name_predict)
        os.remove(path)
        return list_name_predict


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
