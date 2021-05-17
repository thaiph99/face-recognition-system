__author__ = 'thaiph99'

from os import listdir
from flask import Flask, render_template

app = Flask(__name__)


@app.route('/', methods=['GET'])
def page_home():
    hists = listdir('templates/lisa')
    hists = ['lisa/' + file for file in hists]
    return render_template('index1.html', result=hists)


app.run(host='0.0.0.0', port='3000')
