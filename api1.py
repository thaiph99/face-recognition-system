__author__ = 'thaiph99'

import face_recognition
from sklearn import svm
import os
import json
import numpy as np
from numpy.linalg import norm
from json import JSONEncoder
import pickle
from deepface import DeepFace
from deepface.commons import functions, realtime, distance as dst

model_name = "Facenet"
model = DeepFace.build_model(model_name)


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

    def __save_data(self):
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
            # face = face_recognition.load_image_file(filename + '/' + img)
            # face_bounding_boxes = face_recognition.face_locations(
            #     face, model='cnn')

            # detect locations ---------------
            # functions.initialize_detector(detector_backend='mtcnn')
            # face = functions.load_image(filename + '/' + img)
            # face_bounding_boxes = functions.detect_face(
            #     face, detector_backend='mtcnn', enforce_detection=True)[1]
            # if len(face_bounding_boxes) == 1:

            img_path = filename + '/' + img
            face_enc = DeepFace.represent(
                img_path, model_name=model_name, model=model)
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

    @staticmethod
    def is_similarity(unknown_face_em, list_em):
        """
        cal similarity by euclidean distances
        """
        compare = []
        for em in list_em:
            cal_norm2 = norm(em-unknown_face_em)
            compare.append(cal_norm2)
        compare = np.array(compare)
        print(compare)
        return True in (compare <= 0.4)

    def is_known_faces(self, unknown_face_em, list_ems):
        """
        :param unknown_face:
        :return: True if known face else False
        """
        results = []
        for list_em in list_ems:
            list_em = np.array(list_em)
            results.append(self.is_similarity(unknown_face_em, list_em))

        return True in results

    def recognize(self):
        """
        :return: name of old people in image
        """

        dict_em = {}
        # create dict faces
        for i in range(len(self.faces_name)):
            if self.faces_name[i] not in dict_em.keys():
                dict_em[self.faces_name[i]] = []
            dict_em[self.faces_name[i]].append(self.faces_encoded[i])

        # get file name in data temporary
        filename = ''
        filename_dir = self.storage_temporary
        for i in os.listdir(filename_dir):
            if i != '.gitignore':
                filename = i
                break

        path = self.storage_temporary + '/' + filename
        # test_img_encs = DeepFace.represent(
        #     path, model_name=model_name, model=model)
        test_img_encs = DeepFace.represent(
            path, model_name=model_name, model=model)
        test_img_encs = np.array(test_img_encs)

        list_name_predict = []
        result = []
        name = ''
        for key in dict_em.keys():
            list_ems = dict_em[key]
            res = self.is_known_faces(test_img_encs, list_ems)
            result.append(res)

        if True in result:
            name = self.model.predict(test_img_encs)
        else:
            name = ['_unknown face']
        list_name_predict += list(name)
        print(list_name_predict)
        os.remove(path)
        return list_name_predict


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
