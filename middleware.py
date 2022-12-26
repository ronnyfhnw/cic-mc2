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

