# flask imports
from flask import Flask, request, jsonify, render_template, redirect, url_for, make_response
from flask_cors import CORS, cross_origin

# general imports
import json
import requests
from datetime import datetime
import os

# recommender imports
import numpy as np
import pandas as pd

# load dataframe for example responses
df = pd.read_csv('data_tmbd_cleaned.csv', lineterminator='\n', delimiter=';')

##################################################################################################################################################

# load secrets
with open("secrets.json", "r") as f:
    secrets = json.load(f)

# load keys and urls
MIDDLEWARE_KEY = secrets['MIDDLEWARE_KEY']

# start app
app = Flask(__name__)
CORS(app)

@app.route('/')
@cross_origin()
def index():
    return '''<h1>Mock works fine</h1>'''

@app.route("/speech-to-text", methods=['POST'])
@cross_origin()
def speech_to_text():
    if request.method != 'POST' or request.form['key'] != MIDDLEWARE_KEY:
        return {"message": "Forbidden request"}, 403

    if 'audiofile' not in request.files.keys():
        return {"message": "Audiofile missing!"}, 400

    if not request.files['audiofile'].filename.endswith('.wav'):
        return {"message": "Audiofile must be in .wav format!"}, 400

    return jsonify({"text":"random text"})

# start application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5501)