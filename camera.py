__author__ = 'thaiph99'

from imutils.video import WebcamVideoStream
import cv2
import face_recognition


class VideoCamera(object):
    def __init__(self):
        self.stream = WebcamVideoStream(src=0).start()
        # with open()

    def __del__(self):
        self.stream.stop()

    def get_frame(self):
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
