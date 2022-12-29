# flask imports
from flask import Flask, request, jsonify, render_template, redirect, url_for, make_response
from flask_cors import CORS, cross_origin

# general imports
import json
import requests
from datetime import datetime
import os

# speech imports
import azure.cognitiveservices.speech as speechsdk

# recommender imports
import numpy as np
import pandas as pd
from RecommenderSystem import *
from sklearn.metrics.pairwise import pairwise_distances
import torch

##################################################################################################################################################
# load secrets
with open("secrets.json", "r") as f:
    secrets = json.load(f)

# load keys and urls
getEmbeddingUrl = secrets['azure_albert']
watcher_url = secrets['watcher_url']
watcher_auth_token = secrets["watcher_auth_token"]
MIDDLEWARE_KEY = secrets['MIDDLEWARE_KEY']

# config speech2text
speech_config = speechsdk.SpeechConfig(subscription=secrets["S2T_KEY"], region=secrets["S2T_REGION"])
speech_config.speech_recognition_language="en-US"

# start app
app = Flask(__name__)

##################################################################################################################################################

# load data for recommender
descriptions = np.array(torch.load("data/avgpooled_albert_sentence_embeddings_float32.pt", map_location=torch.device("cpu")))
description_matrix = np.array(torch.load("data/pca_albert_description.pt", map_location=torch.device("cpu")))
title_matrix = np.array(torch.load("data/pca_albert_title.pt", map_location=torch.device("cpu")))
cast_matrix = np.array(torch.load("data/pca_albert_cast.pt", map_location=torch.device("cpu")))
df = pd.read_csv("data/data_tmbd_cleaned.csv", delimiter=";", lineterminator="\n")

# filter matrices
description_matrix = description_matrix[list(df.index),:]
descriptions = descriptions[list(df.index),:]
title_matrix = title_matrix[list(df.index),:]
cast_matrix = cast_matrix[list(df.index),:]

# build matrices
movie_info = ContentBasedRecommender.build_movie_info_matrix(df)
mapping_matrix = ContentBasedRecommender.build_mapping_matrix(df)

# build input
matrix_dict = {
        'movie_info_matrix': movie_info,
        'description_matrix': description_matrix,
        'title_matrix': title_matrix,
        'cast_info_matrix': cast_matrix
}

# build recommender system
RS = ContentBasedRecommender(
            mapping_matrix=mapping_matrix,
            **matrix_dict,
            scaling_kwargs=None
        )

# define function for checking balance

def check_balance():
    # send request
    response = requests.get(watcher_url, headers={'auth_token': watcher_auth_token})
    # parse response
    if "true" in response.text:
        return True
    else:
        return False

##################################################################################################################################################

@app.route("/getRecommendationsByIds", methods=['POST'])
def getRecommendationsByIds():
    # get data 
    data = request.json

    if not check_balance():
        return {"message": "Cost limit reached"}, 403

    # check key
    if data['key'] != MIDDLEWARE_KEY:
        return {"message": "Forbidden Request"}, 403

    ids = data['ids']
    assert type(ids) == list

    # build ratings
    ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
    ratings['movieId'] = pd.Series(ids)
    ratings['userId'] = 21
    ratings['rating'] = 5

    # get recommendation 
    recommendation_titles, recommendation_movieIds = RS.get_recommendations_for_user(ratings_user=ratings, n_rec=20)

    # prepare paths, title and description
    response = df[df.movieId.isin(recommendation_movieIds)][['title', 'description', 'poster_path']]
    response = response.to_json(orient="index")

    return response

@app.route("/getRandomMovies", methods=['GET'])
def getRandomMovies():
    # load request data
    data = request.json

    if not check_balance():
        return {"message": "Cost limit reached"}, 403

    # authentication
    if data["key"] != MIDDLEWARE_KEY:
        return {"message": "Forbidden Request"}, 403

    movie_ids = [
        201773, # spider man far from home
        102125, # Iron Man 3
        81834, # harry potter
        111921, # the fault in our stars
        125916, # fifty shades of grey
        136020, # spectre
        160438, # jason borne
        111360, # lucy
        197179, # chaos walking
        175303 # it
        ]

    random_recommendations = df[df.movieId.isin(movie_ids)][['title', 'description', 'poster_path']]
    response = random_recommendations.to_json(orient="index")

    return response

@app.route("/speech2text", methods=['POST'])
def speech2text():
    if not check_balance():
        return {"message": "Cost limit reached"}, 403

    if request.method != 'POST' or request.form['key'] != MIDDLEWARE_KEY:
        return {"message": "Forbidden request"}, 403

    # tmp store file
    save_path = str(datetime.now()) +  "temp.wav"
    request.files['audiofile'].save(save_path)

    # continue processing...
    audio_config = speechsdk.audio.AudioConfig(filename=save_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # get text
    result = speech_recognizer.recognize_once_async().get()
    text = result.text
    
    # delete tmp file
    os.remove(save_path)

    # make request to azure albert
    r = requests.post(getEmbeddingUrl, json={'text':text})
    data = r.json()

    # transform response
    embedding_flattened, shape = data['embedding'], data['shape']
    embedding = torch.tensor(embedding_flattened).reshape(shape)
    
    # calculate cosine similarity between descriptions
    distances = pairwise_distances(descriptions, embedding.reshape(1,768), metric='cosine')
    indices = np.argsort(distances, axis=0).tolist()
    indices = indices[:10]
    
    # get recommendations
    recommendations = pd.DataFrame(columns=df.columns)
    for index in indices:
        recommendations = pd.concat((recommendations, df.iloc[index, :]))

    # recommendations to json
    recommendations = recommendations.reset_index()
    recommendations = recommendations[['title']]
    recommendations = recommendations.to_json(orient="index")

    return recommendations

@app.route("/testSpeech2Text", methods=['GET'])
def testSpeech2Text():
    if not check_balance():
        return {"message": "Cost limit reached"}, 403

    # send audio to speech2text
    audio_config = speechsdk.audio.AudioConfig(filename="test.wav")
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # get text
    result = speech_recognizer.recognize_once_async().get()
    text = result.text

    return text


# start application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5501, debug=True)