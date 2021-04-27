import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pkl
# from keras.models import Sequential, load_model
# from keras.layers import Dense
# from keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


def detect_face_points(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "models/shape_predictor_68_face_landmarks.dat")
    face_rect = detector(image, 1)
    if len(face_rect) != 1:
        return []

    dlib_points = predictor(image, face_rect[0])
    face_points = []
    for i in range(68):
        x, y = dlib_points.part(i).x, dlib_points.part(i).y
        face_points.append(np.array([x, y]))
    return face_points


def compute_features(face_points):
    # assert (len(face_points) == 68), "len(face_points) must be 68"
    if len(face_points) != 68:
        return None
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))

    return np.array(features).reshape(1, -1)


# im = cv2.imread('face-recognition-test/processed_data/train1/Hoang Ngoc/IMG_20200611_104305.jpg', cv2.IMREAD_COLOR)
# im = cv2.imread('datatmp/img_in_sec.jpg', cv2.IMREAD_COLOR)
# im = cv2.imread('face-recognition-test/img1.jpg', cv2.IMREAD_COLOR)
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# face_points = detect_face_points(im)

# for x, y in face_points:
#     cv2.circle(im, (x, y), 1, (0, 255, 0), -1)

# features = compute_features(face_points)
# features = StandardScaler().fit_transform(features)

# model = load_model('models/model.h5')
# y_pred = model.predict(features)

# roll_pred, pitch_pred, yaw_pred = y_pred[0]
# print(' Roll: {:.2f}°'.format(roll_pred))
# print('Pitch: {:.2f}°'.format(pitch_pred))
# print('  Yaw: {:.2f}°'.format(yaw_pred))

# plt.figure(figsize=(10, 10))
# plt.imshow(im)
# plt.show()


camera = cv2.VideoCapture(0)
# model = load_model('models/model.h5')

while True:
    ret, frame = camera.read()
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_points = detect_face_points(im)
    if len(face_points) != 68:
        cv2.imshow('camera', frame)
        continue

    for x, y in face_points:
        print(x, y)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
#     # print('points : ', face_points)
#     features = compute_features(face_points)
#     # features = StandardScaler.fit(features).transform(features.astype='float')
#     print('type : ',  type(features))
#     features = StandardScaler().fit_transform(features)

#     y_pred = model.predict(features)

#     roll_pred, pitch_pred, yaw_pred = y_pred[0]
#     roll = (' Roll: {:.2f}°'.format(roll_pred))
#     pitch = ('Pitch: {:.2f}°'.format(pitch_pred))
#     yaw = ('  Yaw: {:.2f}°'.format(yaw_pred))
#     print(roll)
#     print(pitch)
#     print(yaw)

#     # cv2.putText(frame, roll, (10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
#     #             color=(255, 0, 0), thickness=1)
#     # cv2.putText(frame, pitch, (10, 200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
#     #             color=(255, 0, 0), thickness=1)
#     # cv2.putText(frame, yaw, (10, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
#     #             color=(255, 0, 0), thickness=1)

    cv2.imshow('camera', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
