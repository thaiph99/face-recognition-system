
# coding:utf-8
'''
Face position calibration
'''
import cv2
import dlib
import numpy as np
import time


def shape_to_np(shape, dtype="int"):
    print('type shape :', type(shape))
    coords = np.zeros((5, 2), dtype=dtype)
    for i in range(0, 5):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


predictor_path = 'shape_predictor_5_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("cannot open camear")
    exit(0)

start = time.time()

while True:
    ret, frame = camera.read()
    print('ret : ', ret)
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face
    dets = detector(frame_new, 1)
    for i, det in enumerate(dets):
        left = det.left()
        right = det.right()
        top = det.top()
        bottom = det.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom),
                      color=(0, 0, 0), thickness=5)

    cv2.imshow('camera', frame)

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
    points = shape_to_np(faces)
    print('points : ', points)
    print('len point :', len(points))
    for i in points:
        x = i[0]
        y = i[1]
        cv2.circle(frame, (x, y), 2, color=(0, 0, 0), thickness=-1)
    cv2.imshow('camera', frame)
    image = dlib.get_face_chip(frame_new, faces, size=320)
    cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('face', cv_bgr_img)
    end = time.time()
    cv2.imwrite('data_save/img_in_sec'+str(end-start)+'.jpg', cv_bgr_img)

    print('time : ', 300 - int(end-start))
    if end-start >= 60*3:
        break

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
