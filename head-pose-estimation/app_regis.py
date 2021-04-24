__author__ = 'thaiph99'

from flask import Flask, json, Response, request, render_template
from werkzeug.utils import secure_filename
from os import path, getcwd
import time
from collections import Counter
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Route for Homepage
@app.route('/', methods=['GET'])
def page_home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='localhost', port=3000)
