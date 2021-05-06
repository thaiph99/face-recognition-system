from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image
import mtcnn
import Facenet
import matplotlib.pyplot as plt
from matplotlib import pyplot
from os import listdir
from os.path import isdir
from numpy import savez_compressed

# turn off gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print('mtcnn version :', mtcnn.__version__)

# model = Facenet.loadModel()
# load the model

# summarize input and output shape
# print(model.inputs)
# print(model.outputs)

# function for face detection with mtcnn

# extract a single face from a given photograph


def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


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

# load the photo and extract the face
# pixels = extract_face('messi.jpg')
# plt.imshow(pixels)
# plt.show()

# folder = 'faces/datatrain/Hoang Ngoc/'
# i = 1

# for filename in listdir(folder):
#     # path
#     path = folder + filename
#     # get face
#     face = extract_face(path)
#     print(i, face.shape)
#     # plot
#     pyplot.subplot(2, 7, i)
#     pyplot.axis('off')
#     pyplot.imshow(face)
#     i += 1
# pyplot.show()


# load images and extract faces for all images in a directory
def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces

# load a dataset that contains one subdir for each class that in turn contains images


def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        print(path)
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


# load train dataset
trainX, trainy = load_dataset('faces/datatrain/')
print(len(trainX))
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('faces/datatest/')
# save arrays to one file in compressed format
savez_compressed('face.npz', trainX, trainy, testX, testy)
