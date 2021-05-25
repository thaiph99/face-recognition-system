__author__ = 'thaiph99'

from os import listdir
from flask import Flask, render_template, send_from_directory

app = Flask(__name__)


@app.route('/<filename>')
def send_image(filename):
    return send_from_directory("templates/lisa", filename)


@app.route('/test', methods=['GET'])
def page_home():
    hists = listdir('templates/lisa')
    hists = ['/' + file for file in hists]
    print(hists)
    return render_template('index1.html', result=hists)


app.run(host='0.0.0.0', port='3000', debug=True)
