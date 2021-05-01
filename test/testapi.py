from deepface import DeepFace
from deepface.commons import functions, realtime, distance as dst
import matplotlib.pyplot as plt
import os
import numpy as np


def test1(file_name):
    # filename = self.storage_temporary
    # pix = os.listdir(file_name)

    # for img in pix:
    # face = face_recognition.load_image_file(filename + '/' + img)
    functions.initialize_detector(detector_backend='mtcnn')
    face = functions.load_image(file_name)
    # face_bounding_boxes = face_recognition.face_locations(
    #     face, model='cnn')

    img, locations = functions.detect_face(
        face, detector_backend='mtcnn', enforce_detection=True)
    return img, locations


img_path = '/home/thai/Data2/face-recognition-system/testcode/datatest/Nayeon.jpg'

img, locations = test1(img_path)
print('info img :', type(img), img.shape)
print('info location :', type(locations), len(locations))
print(locations)
# plt.imshow(img)
# plt.show()

# model = DeepFace.build_model('Facenet')
# embedding = model.predict(img)[0].tolist()
# print(type(embedding))
# print(embedding.shape)

embed = DeepFace.represent(img_path, model_name='Facenet')
print(type(embed))
print(len(embed))
print(embed)

# DeepFace.find(img_path, db_path)
embed1 = DeepFace.represent(img_path, model_name='VGG-Face')
print(type(embed1))
print(len(embed1))
print(embed1)


embed2 = DeepFace.represent(img_path, model_name='Facenet')
print(type(embed2))
print(len(embed2))
print(embed2)
# print(sum(embed2))

em = [embed, embed1, embed2]
em = [np.array(e) for e in em]

print('sum 1 : ', np.sum(em[2]-em[0]))
# print('sum 2 : ', np.sum(em[1]-em[0]))

print('sum one embeb: ', np.sum(em[0]))
print('sum one embeb: ', np.sum(em[1]))
print('sum one embeb: ', np.sum(em[2]))
