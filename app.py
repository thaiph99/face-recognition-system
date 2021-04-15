from flask import Flask, json, Response, request, render_template
from werkzeug.utils import secure_filename
from os import path, getcwd
import time
from face import Face

app = Flask(__name__)

app.config['file_allowed'] = ['image/png', 'image/jpeg', 'image/jpg']
app.config['storage'] = path.join(getcwd(), 'storage')
app.config['storage_temporary'] = path.join(getcwd(), 'storage_temporary')
app.face = Face(app)


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


@app.route('/api/train', methods=['POST'])
def train():
    name = request.form['name']
    uploaded_files = request.files.getlist("file[]")
    if len(uploaded_files) > 0:
        print('upload done')
    else:
        print('upload failed')

    for file in uploaded_files:
        file.save(path.join(app.config['dir'], file.filename))


    # if 'file' not in request.files:
    #     print("Face image is required")
    #     return error_handle("Face image is required.")
    # else:
    #     print("File request", request.files)
    #     file = request.files['file']
    #
    #     if file.mimetype not in app.config['file_allowed']:
    #         print("File extension is not allowed")
    #         return error_handle("We are only allow upload file with *.png , *.jpg")
    #     else:
    #         name = request.form['name']
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
            unknown_storage = path.join(app.config["storage"], 'unknown')
            file_path = path.join(unknown_storage, filename)
            file.save(file_path)

            user_id = app.face.recognize(filename)
            if user_id:
                user = get_user_by_id(user_id)
                message = {"message": "Hey we found {0} matched with your face image".format(user["name"]),
                           "user": user}
                return success_handle(json.dumps(message))
            else:
                return error_handle("Sorry we can not found any people matched with your face image, try another image")


# Run the app
app.run()
