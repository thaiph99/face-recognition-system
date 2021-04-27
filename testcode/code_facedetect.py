# coding:utf-8
'''
Face position calibration
'''
import cv2
import dlib
import numpy as np


def shape_to_np(shape, dtype="int"):
    print('type shape :', type(shape))
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 5):
        coords[i] = (shape.parts(i).x, shape.parts(i).y)
    return coords


predictor_path = 'shape_predictor_5_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("cannot open camear")
    exit(0)

while True:
    ret, frame = camera.read()

    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    cv2.imshow('camera', frame)
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face
    dets = detector(frame_new, 1)
    print('dets : ', type(dets))
    # cv2.rectangle(frame_new, dets[0][0], dets[0][1], color=(0, 0, 0))
    print("Number of faces detected: {}".format(len(dets)))
    num_faces = len(dets)
    if num_faces != 1:
        print("Sorry, you can register with only your face")
        continue
        # Find the location of the face
    faces = dlib.full_object_detections()
    # for detection in dets:
    faces = sp(frame_new, dets[0])
    print('type : ', type(faces))
    print(faces)
    # points = []
    # for face in faces:
    #     points.append(shape_to_np(faces))
    points = shape_to_np(faces)
    for i in points:
        x = i[0]
        y = i[1]
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    image = dlib.get_face_chip(frame_new, faces, size=320)
    # for image in images:
    cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('image', cv_bgr_img)
    # image = dlib.get_face_chip(frame_new, faces[0])
    # cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('image', cv_bgr_img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
