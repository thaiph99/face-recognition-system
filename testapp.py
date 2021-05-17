__author__ = 'thaiph99'

from flask import Flask, json, Response, request, render_template
from werkzeug.utils import secure_filename
from os import path, getcwd
import time
from collections import Counter
from flask_cors import CORS
from camera import VideoCamera
from os import listdir
import shutil

app = Flask(__name__)


@app.route('/', methods=['GET'])
def page_home():
    hists = listdir('templates/lisa')
    hists = ['lisa/' + file for file in hists]
    return render_template('index1.html', result=hists)


app.run(host='0.0.0.0', port='3000')
