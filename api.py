__author__ = 'thaiph99'

import json
import os
from json import JSONEncoder
from math import log, e
import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from numpy import asarray, expand_dims
from numpy.linalg import norm
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import facenet
import cv2

# turn off gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = facenet.loadModel()
model.summary()
print('name model :', model.name)


# visualkeras.layered_view(model, to_file='output.png')


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
        self.model_svm = svm.SVC(kernel='linear', probability=True)
        # self.model_svm = pickle.load(open(filename, 'rb'))
        self.model_svm = joblib.load(filename)

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

    def train_again(self):
        self.model_svm = svm.SVC(kernel='linear', probability=True)
        self.model_svm.fit(self.faces_embedded, self.faces_name)
        self.__save_model()
        self.__save_data()

    @staticmethod
    def extract_multi_face(filename, required_size=(160, 160)):
        img = Image.open(filename)
        image = img.convert('RGB')
        pixels = asarray(image)
        img_draw = pixels.copy()
        detector = MTCNN()
        results = detector.detect_faces(pixels)
        face_arrays = []
        for res in results:
            x1, y1, width, height = res['box']
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (225, 225, 0), 2)
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            face = Image.fromarray(face)
            face = face.resize(required_size)
            face_array = asarray(face)
            face_arrays.append(face_array)
            # plt.imshow(face_array)
            # plt.show()
        img_save = Image.fromarray(img_draw)
        img_save.save('dataset/datasave/A.jpg')
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

            # face_enc = bla bla -> face_array  # embedding for face image
            face_enc = self.get_embedding(model, face_array)

            self.faces_embedded.append(face_enc)
            self.faces_name.append(name)
            # clear storage temporary
            os.remove(filename + '/' + img)

    def __save_model(self):
        filename = self.storage + '/' + 'svm_model.sav'
        # pickle.dump(self.model_svm, open(filename, 'wb'))
        joblib.dump(self.model_svm, filename)

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
        faces_em = np.array(self.faces_embedded)

        in_encoder = Normalizer(norm='l2')
        # print('shape ', faces_em.shape)
        face_embeded_encoded = in_encoder.transform(faces_em)
        face_name_encoded = out_encoder.transform(self.faces_name)
        # save out encoder
        np.save(self.storage + '/' + 'face_name_encoded.npy',
                out_encoder.classes_)
        self.__save_data()
        self.model_svm = svm.SVC(kernel='linear', probability=True)
        self.model_svm.fit(face_embeded_encoded, face_name_encoded)
        self.__save_model()

        return 1

    @staticmethod
    def cal_similarity_score(embed, name, dict_face):
        embed = np.array(embed)
        list_norm = []
        for em in dict_face[name]:
            em = np.array(em)
            # print('em :', em.shape, sum(em))
            # print('embed:', embed.shape, sum(embed))
            list_norm.append(norm(em - embed, ord=2))
        print('list norm', list_norm)
        return sum(list_norm) / len(list_norm)

    @staticmethod
    def is_similarity(unknown_face_em, list_em):
        """
        cal similarity by norm 
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

    def get_embeb_multi_faces(self, path):
        faces_test = self.extract_multi_face(path)

        # get embedding all faces
        test_img_encs = []
        for face_fixel in faces_test:
            test_img_encs.append(self.get_embedding(model, face_fixel))
        test_img_encs = np.array(test_img_encs)

        in_encoder = Normalizer(norm='l2')
        test_img_encs_origin = test_img_encs
        test_img_encs = in_encoder.transform(test_img_encs)
        return test_img_encs_origin, test_img_encs, faces_test

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
        test_img_encs_origin, test_img_encs, faces_test = self.get_embeb_multi_faces(
            path)

        list_name_predict = []
        result = []
        name = ''
        for key in dict_em.keys():
            list_ems = dict_em[key]
            res = self.is_known_faces(test_img_encs, list_ems)
            result.append(res)
        # test True
        out_encoder = LabelEncoder()
        out_encoder.classes_ = np.load(
            self.storage + '/' + 'face_name_encoded.npy')
        print('info model ', self.model_svm)
        face_class = self.model_svm.predict(test_img_encs)
        face_prob = self.model_svm.predict_proba(test_img_encs)
        print('class : ', face_class)
        print('out encoder classes : ', out_encoder.classes_)
        name = out_encoder.inverse_transform(face_class)

        dict_name = {}
        for i in range(len(name)):
            similarity_score = self.cal_similarity_score(
                test_img_encs_origin[i], name[i], dict_em)
            print('------------------')
            print('similarity', similarity_score)
            print('name ', name[i])
            if name[i] not in dict_name.keys():
                dict_name[name[i]] = [similarity_score, i]
            else:
                print(name[i])
                print(dict_name[name[i]])
                if dict_name[name[i]][0] > similarity_score:
                    dict_name[name[i]][0] = similarity_score
                    name[dict_name[name[i]][1]] = 'unknown'+str(i)
                    dict_name[name[i]][1] = i
                else:
                    name[i] = 'unknown'+str(i)

            if similarity_score >= 10:
                name[i] = 'unknown'+str(i)

        # fg, ax = plt.subplots(1, len(name))

        for i in range(len(name)):
            print('-----------------')
            print('name', name[i])
            print('probability', face_prob[i])
            print('entropy', self.entropy(face_prob[i]))
            print('type ', type(faces_test[i]))
            img_save = Image.fromarray(faces_test[i])
            img_save.save(f'dataset/datasave/{name[i]}.jpg')
            # ax[i].imshow(faces_test[i])
            # title = (str(name[i]))
            # ax[i].set_title(title)
            # ax[i].axis('off')
        # plt.show()
        list_name_predict += list(name)
        print('list name', list_name_predict)
        os.remove(path)
        return list_name_predict


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
