import face_recognition
import cv2
import matplotlib.pyplot as plt

image = face_recognition.load_image_file("datatest/testimg.jpg")

face_locations = face_recognition.face_locations(image)
print(type(face_locations))
print(len(face_locations))
for location in face_locations:
    start_point = (location[0], location[3])
    end_point = (location[1], location[2])
    color = (255, 0, 0)
    thickness = 2
    cv2.rectangle(image, start_point, end_point, color, thickness)

plt.imshow(image)
plt.show()
# cv2.imshow('win img', image)  
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# from PIL import Image, ImageDraw
# import face_recognition

# # Load the jpg file into a numpy array
# image = face_recognition.load_image_file("testimg.jpg")

# # Find all facial features in all the faces in the image
# face_landmarks_list = face_recognition.face_landmarks(image)

# pil_image = Image.fromarray(image)
# for face_landmarks in face_landmarks_list:
#     d = ImageDraw.Draw(pil_image, 'RGBA')

#     # Make the eyebrows into a nightmare
#     d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
#     d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
#     d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
#     d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

#     # Gloss the lips
#     d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
#     d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
#     d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
#     d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

#     # Sparkle the eyes
#     d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
#     d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

#     # Apply some eyeliner
#     d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
#     d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

# pil_image.show()
