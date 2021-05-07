__author__ = 'thaiph99'

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
# turn off gpu
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = Facenet.loadModel()


class Model:
    def __init__(self):
        # for training
        self.storage_temporary = 'storage_temporary'
        self.storage = 'storage'
        self.faces_embedded = []
        self.faces_name = []
        # for recognition
        self.model_svm = None

        self.__load_model()
        self.__load_data()

    def __load_model(self):
        filename = self.storage + '/' + 'svm_model.sav'
        self.model_svm = pickle.load(open(filename, 'rb'))

    def __load_data(self):
        filename = self.storage + '/' + 'face_data.json'
        with open(filename, 'r') as f:
            data = json.load(f)
        self.faces_embedded = []
        self.faces_name = []
        for name in data.keys():
            for fa in data[name]:
                self.faces_embedded.append(fa)
                self.faces_name.append(name)

    def __save_data(self):
        filename = self.storage + '/' + 'faces_data.json'
        with open('face_data.json', 'w') as f:
            json.dump(filename, f, cls=NumpyArrayEncoder)

    def delete_face(self, name_del):
        json_file = {}
        for name in self.faces_name:
            if name == name_del:
                continue
            json_file[name] = []
        for i in range(len(self.faces_embedded)):
            if self.faces_name[i] == name_del:
                continue
            json_file[self.faces_name[i]].append(self.faces_embedded[i])

        filename = self.storage + '/' + 'face_data.json'
        with open(filename, 'w') as f:
            json.dump(json_file, f, cls=NumpyArrayEncoder)
        self.__load_model()
        self.__load_data()
        self.train_again()

    @staticmethod
    def extract_multi_face(filename, required_size=(160, 160)):
        image = Image.open(filename)
        image = image.convert('RGB')
        pixels = asarray(image)
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        face_arrays = []
        for res in results:
            x1, y1, width, height = res['box']
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image)
            face_arrays.append(face_array)
            # plt.imshow(face_array)
            # plt.show()
        return face_arrays

    @staticmethod
    def get_embedding(model, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]

    def __append_new_face(self, name):
        filename = self.storage_temporary
        pix = os.listdir(filename)
        for img in pix:
            if img == '.gitignore':
                continue

            img_path = filename + '/' + img

            image = Image.open(img_path)
            image = image.convert('RGB')
            pixels = asarray(image)
            detector = MTCNN()
            results = detector.detect_faces(pixels)
            if len(results) != 1:
                continue

            x1, y1, width, height = results[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            image = Image.fromarray(face)
            image = image.resize((160, 160))
            face_array = asarray(image)

            # face_enc = bla bla -> face_array  # embeding for face image
            face_enc = self.get_embedding(model, face_array)

            self.faces_embedded.append(face_enc)
            self.faces_name.append(name)
            # clear storage temporary
            os.remove(filename + '/' + img)

    def __save_model(self):
        filename = self.storage + '/' + 'svm_model.sav'
        pickle.dump(self.model_svm, open(filename, 'wb'))

    def __save_data(self):
        json_file = {}
        for name in self.faces_name:
            json_file[name] = []
        for i in range(len(self.faces_embedded)):
            json_file[self.faces_name[i]].append(self.faces_embedded[i])

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
        out_encoder = LabelEncoder()
        out_encoder.fit(self.faces_name)
        face_name_encoded = out_encoder.transform(self.faces_name)
        # save out encoder
        np.save(self.storage + '/' + 'face_name_encoded.npy',
                out_encoder.classes_)
        self.__save_data()
        self.model_svm = svm.SVC(kernel='linear', probability=True)
        self.model_svm.fit(self.faces_embedded, face_name_encoded)
        self.__save_model()

        return 1

    # def train_again(self):
    #     self.model_svm = svm.SVC(gamma='scale')
    #     self.model_svm.fit(self.faces_embedded, self.faces_name)
    #     self.__save_model()
    #     self.__save_data()

    @staticmethod
    def is_similarity(unknown_face_em, list_em):
        """
        cal similarity by euclidean distances
        """
        compare = []
        for em in list_em:
            cal_norm2 = norm(em - unknown_face_em)
            compare.append(cal_norm2)
        compare = np.array(compare)
        # print(compare)
        return True in (compare <= 0.4)

    @staticmethod
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

    def is_known_faces(self, unknown_face_em, list_ems):
        """
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
            dict_em[self.faces_name[i]].append(self.faces_embedded[i])

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

        faces_test = self.extract_multi_face(path)

        test_img_encs = []
        for face_fixel in faces_test:
            print(face_fixel.shape)
            test_img_encs.append(self.get_embedding(model, face_fixel))
        test_img_encs = np.array(test_img_encs)

        list_name_predict = []
        result = []
        name = ''
        for key in dict_em.keys():
            list_ems = dict_em[key]
            res = self.is_known_faces(test_img_encs, list_ems)
            result.append(res)
        # test trua
        result.append(True)
        if True in result:
            out_encoder = LabelEncoder()
            out_encoder.classes_ = np.load(
                self.storage + '/' + 'face_name_encoded.npy')
            face_class = self.model_svm.predict(test_img_encs)
            face_prob = self.model_svm.predict_proba(test_img_encs)
            name = out_encoder.inverse_transform(face_class)
            for i in range(len(name)):
                print('name', name[i])
                print('probability', face_prob[i]*100)
                print('entropy', self.entropy(face_prob[i]))
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
