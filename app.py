__author__ = 'thaiph99'

from flask import Flask, json, Response, request, render_template
from werkzeug.utils import secure_filename
from os import path, getcwd
import time
from api import Model
from collections import Counter
from flask_cors import CORS
from camera import VideoCamera
from os import listdir
import shutil

app = Flask(__name__)
CORS(app)

app.config['file_allowed'] = ['image/png', 'image/jpeg', 'image/jpg']
app.config['storage'] = path.join(getcwd(), 'storage')
app.config['storage_temporary'] = path.join(getcwd(), 'storage_temporary')
app.model = Model()


def success_handle(output, status=200, mimetype='application/json'):
    return Response(output, status=status, mimetype=mimetype)


def error_handle(error_message, status=500, mimetype='application/json'):
    return Response(json.dumps({"error": {"message": error_message}}), status=status, mimetype=mimetype)


#   Route for Homepage
@app.route('/', methods=['GET'])
def page_home():
    return render_template('index.html')


@app.route('/api', methods=['GET'])
def homepage():
    output = json.dumps({"api": '1.0'})
    return success_handle(output)


def gen(camera):
    start = time.time()
    while True:
        end = time.time()
        data = camera.get_frame(60 - int(end - start))
        # data = camera.get_frame_by_face_recognition()
        if int(end - start) >= 60:
            break
        frame = data
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/regis')
def regis():
    print('Camera action')
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/train', methods=['POST'])
def train():
    name = request.form['name']
    uploaded_files = request.files.getlist("file[]")

    check_data = 0

    list_img_tmp = listdir('data_face_temporary')
    # print(list_img_tmp)
    if len(list_img_tmp) != 1:
        check_data = 1
        for filename in list_img_tmp:
            if filename == '.gitignore':
                continue
            shutil.move("data_face_temporary/" + filename,
                        "storage_temporary/" + filename)

    if (len(uploaded_files) == 0 or name in set(app.model.faces_name)) and check_data != 1:
        print(len(uploaded_files))
        print()
        print('upload image failed')
        check_data = 0
    else:
        for file in uploaded_files:
            file.save(
                path.join(app.config['storage_temporary'], file.filename))
        check_data = 1

    if check_data == 0:
        return error_handle('upload failed')

    app.model.train(name)
    print('upload done')
    return success_handle('accepted')


# router for recognize a unknown face
@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return error_handle("Image is required")
    else:
        file = request.files['file']
        # file extension validate
        if file.mimetype not in app.config['file_allowed']:
            return error_handle("File extension is not allowed")
        else:
            filename = secure_filename(file.filename)
            # filename = '/img_test.jpg'
            # unknown_storage = path.join(app.config["storage"], 'unknown')
            file_path = path.join(app.config['storage_temporary'], filename)
            file.save(file_path)
            print(file_path)
            list_name = app.model.recognize()
            if len(list_name) != 0:
                user_name = list_name
                message = {"message": "Hey we found {0} matched with your face image".format(len(user_name)),
                           "user": user_name}
                return success_handle(json.dumps(message))
            else:
                return error_handle("Sorry we can not found any people matched with your face image, try another image")


# app.model.delete_face('lisa')
set_name = Counter(app.model.faces_name)
print(set_name)

# Run the app
app.run(host='0.0.0.0', port='3000')
# app.run(host='localhost', port=3000)
