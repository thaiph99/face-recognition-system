from deepface import DeepFace
from deepface.commons import functions, realtime, distance as dst
import matplotlib.pyplot as plt
import os
import numpy as np

# DeepFace.verify(img1_path)


def test1(file_name):

    functions.initialize_detector(detector_backend='mtcnn')
    face = functions.load_image(file_name)

    img, locations = functions.detect_face(
        face, detector_backend='mtcnn', enforce_detection=True)
    return img, locations


img_path = 'face-recognition-test/processed_data/train/Lisa/Screenshot_20181028-165521__01.jpg'

img, locations = test1(img_path)
print('info img :', type(img), img.shape)
print('info location :', type(locations), len(locations))
print(locations)
# plt.imshow(img)
# plt.show()image_face = image[y1: y2, x1: x2]

# model = DeepFace.build_model('Facenet')
# embedding = model.predict(img)[0].tolist()
# print(type(embedding))
# print(embedding.shape)

# embed = DeepFace.represent(img_path, model_name='Facenet')
# print(type(embed))
# print(len(embed))
# print(embed)

# DeepFace.find(img_path, db_path)
# embed1 = DeepFace.represent(img_path, model_name='VGG-Face')
# print(type(embed1))
# print(len(embed1))
# print(embed1)


# embed2 = DeepFace.represent(img_path, model_name='Facenet')
# print(type(embed2))
# print(len(embed2))
# print(embed2)
# print(sum(embed2))

# em = [embed, embed1, embed2]
# em = [np.array(e) for e in em]

# print('sum 1 : ', np.sum(em[2]-em[0]))
# print('sum 2 : ', np.sum(em[1]-em[0]))

# print('sum one embeb: ', np.sum(em[0]))
# print('sum one embeb: ', np.sum(em[1]))
# print('sum one embeb: ', np.sum(em[2]))
