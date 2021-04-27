__author__ = 'thaiph99'
import cv2
import math
import numpy as np


def face_orientation(frame, landmarks):
    size = frame.shape  # (height, width, color_channel)

    image_points = np.array([
                            (landmarks[4], landmarks[5]),     # Nose tip
                            (landmarks[10], landmarks[11]),   # Chin
                            # Left eye left corner
                            (landmarks[0], landmarks[1]),
                            # Right eye right corne
                            (landmarks[2], landmarks[3]),
                            # Left Mouth corner
                            (landmarks[6], landmarks[7]),
                            # Right mouth corner
                            (landmarks[8], landmarks[9])
                            ], dtype="double")

    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            # Right eye right corne
                            (165.0, 170.0, -135.0),
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    # Camera internals

    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.CV_ITERATIVE)

    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    imgpts, jac = cv2.projectPoints(
        axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(
        model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[4], landmarks[5])
