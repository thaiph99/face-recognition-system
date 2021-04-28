__author__ = 'thaiph99'

from imutils.video import WebcamVideoStream
import cv2
import face_recognition
import dlib
import numpy as np


# predictor_path = 'testcode/shape_predictor_5_face_landmarks.dat'


class VideoCamera(object):
    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()
        self.detertor = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            'testcode/shape_predictor_5_face_landmarks.dat')
        # with open()

    def __del__(self):
        self.stream.stop()

    def get_frame_by_face_recognition(self):
        img = self.stream.read()
        img = cv2.flip(img, 1)

        locations = face_recognition.face_locations(img)

        for top, right, bottom, left in locations:
            startX = int(left)
            startY = int(top)
            endX = int(right)
            endY = int(bottom)
            cv2.rectangle(img, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)

        _, jpg = cv2.imencode('.jpg', img)
        jpg = jpg.tobytes()
        return jpg

    @staticmethod
    def shape_to_np(shape, dtype="int"):
        # print('type shape :', type(shape))
        coords = np.zeros((5, 2), dtype=dtype)
        for i in range(0, 5):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def get_frame(self, time):
        frame = self.stream.read()
        frame = cv2.flip(frame, 1)
        frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = self.detertor(frame_new, 1)
        if len(dets) == 1:
            for i, det in enumerate(dets):
                startX = int(det.left())
                startY = int(det.top())
                endX = int(det.right())
                endY = int(det.bottom())
                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), (0, 255, 0), 2)

            faces = dlib.full_object_detections()
            faces = self.predictor(frame_new, dets[0])

            points = self.shape_to_np(faces)

            for i in points:
                x = i[0]
                y = i[1]
                cv2.circle(frame, (x, y), 2, color=(0, 0, 0), thickness=2)

            ratio1 = abs(points[1][0]-points[4][0])
            ratio2 = abs(points[3][0]-points[4][0])
            if ratio2 == 0 or ratio1 == 0:
                ratio = 0
            elif ratio1 > ratio2:
                ratio = ratio1/ratio2
            else:
                ratio = -ratio2/ratio1
            print(ratio)
            list_left_face = [1, 2, 3, 4, 5, 6, 7, 8]
            list_right_face = [-1, -2, -3, -4, -5, -6, -7, -8]
            if ratio > 0:
                recomend = 'HAY QUAY SANG PHAI'
            else:
                recomend = 'HAY QUAY SANG TRAI'
            cv2.putText(frame, str(ratio), org=(
                10, 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 0, 0))
            cv2.putText(frame, recomend, org=(
                10, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 0, 0))
            cv2.putText(frame, f'time :{time}', org=(
                10, 90), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 0, 0))

            image = dlib.get_face_chip(frame_new, faces, size=320)
            cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite('datatmp/img_in_sec.jpg', cv_bgr_img)
        _, jpg = cv2.imencode('.jpg', frame)
        jpg = jpg.tobytes()
        return jpg

    def get_frame1(self, time):
        frame = self.stream.read()
        frame = cv2.flip(frame, 1)
        frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = face_recognition.face_locations(frame_new)
        if len(dets) == 1:
            # for i, det in enumerate(dets):
            #     startX = int(det.left())
            #     startY = int(det.top())
            #     endX = int(det.right())
            #     endY = int(det.bottom())
            #     cv2.rectangle(frame, (startX, startY),
            #                   (endX, endY), (0, 255, 0), 2)

            for det in dets:
                startX = int(det[3])
                startY = int(det[0])
                endX = int(det[1])
                endY = int(det[2])
                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), (0, 255, 0), 2)

            # faces = dlib.full_object_detections()
            # faces = self.predictor(frame_new, dets[0])

            # points = self.shape_to_np(faces)
            points = face_recognition.face_landmarks(frame, dets)

            for i in points:
                x = i[0]
                y = i[1]
                cv2.circle(frame, (x, y), 2, color=(0, 0, 0), thickness=2)

            ratio1 = abs(points[1][0]-points[4][0])
            ratio2 = abs(points[3][0]-points[4][0])
            if ratio2 == 0 or ratio1 == 0:
                ratio = 0
            elif ratio1 > ratio2:
                ratio = ratio1/ratio2
            else:
                ratio = -ratio2/ratio1
            print(ratio)
            list_left_face = [1, 2, 3, 4, 5, 6, 7, 8]
            list_right_face = [-1, -2, -3, -4, -5, -6, -7, -8]
            if ratio > 0:
                recomend = 'HAY QUAY SANG PHAI'
            else:
                recomend = 'HAY QUAY SANG TRAI'
            cv2.putText(frame, str(ratio), org=(
                10, 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 0, 0))
            cv2.putText(frame, recomend, org=(
                10, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 0, 0))
            cv2.putText(frame, f'time :{time}', org=(
                10, 90), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(255, 0, 0))

            image = dlib.get_face_chip(frame_new, faces, size=320)
            cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite('datatmp/img_in_sec.jpg', cv_bgr_img)
        _, jpg = cv2.imencode('.jpg', frame)
        jpg = jpg.tobytes()
        return jpg
