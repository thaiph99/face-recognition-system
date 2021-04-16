import face_recognition
import cv2

image = face_recognition.load_image_file("datatest/testimg.jpg")
face_locations = face_recognition.face_locations(image)

for location in face_locations:
    start_point = location[:2]
    end_point = location[2:]
    color = (255, 0, 0)
    thickness = 2
    cv2.rectangle(image, start_point, end_point, color, thickness)

cv2.imshow('win img', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(*face_locations, sep='\n')
