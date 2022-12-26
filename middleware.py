from flask import Flask, request, jsonify, render_template, redirect, url_for, make_response
from flask_cors import CORS, cross_origin

import json
from datetime import datetime
import requests
import os

import math
import base64
import numpy as np
import io

# load env
with open('secrets.json', 'r') as f:
    secrets = json.load(f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

# start application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5500, debug=True)