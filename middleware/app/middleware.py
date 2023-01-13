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
speech_config = speechsdk.SpeechConfig(
    subscription=secrets["S2T_KEY"], region=secrets["S2T_REGION"])
speech_config.speech_recognition_language = "en-US"

# start app
app = Flask(__name__)
CORS(app)

##################################################################################################################################################

# load data for recommender
descriptions = np.array(torch.load(
    "data/avgpooled_albert_sentence_embeddings_float32.pt", map_location=torch.device("cpu")))
description_matrix = np.array(torch.load(
    "data/pca_albert_description.pt", map_location=torch.device("cpu")))
title_matrix = np.array(torch.load(
    "data/pca_albert_title.pt", map_location=torch.device("cpu")))
cast_matrix = np.array(torch.load(
    "data/pca_albert_cast.pt", map_location=torch.device("cpu")))
# df = pd.read_csv("data/data_tmbd_cleaned.csv", delimiter=";", lineterminator="\n")
df = pd.read_pickle("data/df.pkl")

# filter matrices
# description_matrix = description_matrix[list(df.index),:]
# descriptions = descriptions[list(df.index),:]
# title_matrix = title_matrix[list(df.index),:]
# cast_matrix = cast_matrix[list(df.index),:]

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
    response = requests.get(watcher_url + "?key=" + watcher_auth_token)
    # parse response
    if "true" in response.text:
        return True
    else:
        return False


# prepare dataframe and function for drawing samples
# define top genres
top_genres = ['Adventure', 'Animation', 'Fantasy', 'Action',
              'Sci-Fi', 'Horror', 'Thriller', 'Romance', 'Drama', 'Comedy']

# get most rated movies
top_movies = df.sort_values(by='vote_count', ascending=False).head(1000)


def draw_samples(top_movies: pd.DataFrame, top_genres: list) -> pd.DataFrame:
    '''
    This function draws 15 samples from the top 1000 most-rated movies. Ten samples are drawn from the most popular genres, five more at random. 

    Args:
    -----
        top_movies:pd.DataFrame - A pandas dataframe containing the top 1000 most-rated movies
        top_genres:list - List containing all genres to select from

    Returns:
    --------
        random_samples:pd.DataFrame - A pandas dataframe containing fifteen samples

    '''
    # prepare dataframe for samples
    random_samples = pd.DataFrame(columns=top_movies.columns)

    # loop over genres to draw samples
    for genre in top_genres:
        genre_movies = top_movies[top_movies[genre] == 1]
        drawn_sample = genre_movies.sample(n=1)

        # check if sample not already in samples
        while random_samples[random_samples.title == drawn_sample.title.values[0]].shape[0] != 0:
            drawn_sample = genre_movies.sample(n=1)

        random_samples = pd.concat((random_samples, drawn_sample))

    # add five more random samples
    random_samples = pd.concat((random_samples, top_movies.sample(n=5)))

    # postprocess titles
    random_samples.title = random_samples.title.apply(
        lambda x: "The " + x.rstrip(", The") if x.endswith(", The") else x)
    random_samples.title = random_samples.title.apply(
        lambda x: "A " + x.rstrip(", A") if x.endswith(", The") else x)

    return random_samples

##################################################################################################################################################


@app.route('/')
@cross_origin()
def index():
    return '''<h1>Middleware works fine</h1>'''


@app.route("/getRecommendationsByIds", methods=['POST'])
@cross_origin()
def getRecommendationsByIds():
    # get data
    data = request.json

    if not check_balance():
        return {"message": "Cost limit reached"}, 403

    # check key
    if data['key'] != MIDDLEWARE_KEY:
        return {"message": "Forbidden Request"}, 403

    ids = data['ids']
    if type(ids) != list:
        return {"message": "Forbidden Request"}, 403

    # build ratings
    ratings = pd.DataFrame(columns=['userId', 'movieId', 'rating'])
    ratings['movieId'] = pd.Series(ids)
    ratings['userId'] = 21
    ratings['rating'] = 5

    # get recommendation
    recommendation_titles, recommendation_movieIds = RS.get_recommendations_for_user(
        ratings_user=ratings, n_rec=20)

    # prepare paths, title and description
    response = df[df.movieId.isin(recommendation_movieIds)]

    # turn dummy encoding back
    genres = []

    for i in range(response.shape[0]):
        df_tmp = response.iloc[i, :].T[['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action',
                                        'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'IMAX', 'Documentary', 'War', 'Musical', 'Western', 'Film-Noir']]
        genres.append(list(df_tmp[df_tmp == 1].index))

    response['genres'] = pd.Series(genres)

    # filter columns
    response = response[['title', 'description', 'poster_path',
                         'vote_average', 'actor1', 'actor2', 'actor3', 'year', 'genres']]
    response = response.to_json(orient="index")
    return response


@app.route("/getRandomMovies", methods=['GET'])
@cross_origin()
def getRandomMovies():
    # check balance
    if not check_balance():
        return {"message": "Cost limit reached"}, 403

    print(request.args)
    # authentication
    if request.args.get("key") != MIDDLEWARE_KEY:
        return {"message": "Forbidden Request"}, 403

    # draw samples
    random_recommendations = draw_samples(
        top_movies=top_movies, top_genres=top_genres)

    # turn dummy encoding back
    genres = []

    for i in range(random_recommendations.shape[0]):
        df_tmp = random_recommendations.iloc[i, :].T[['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama',
                                                      'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'IMAX', 'Documentary', 'War', 'Musical', 'Western', 'Film-Noir']]
        genres.append(list(df_tmp[df_tmp == 1].index))

    # add genres as column to df
    random_recommendations = random_recommendations.copy()
    random_recommendations.loc[:, 'genres'] = pd.Series(genres)

    # filter columns
    random_recommendations = random_recommendations[[
        'title', 'description', 'poster_path', 'vote_average', 'actor1', 'actor2', 'actor3', 'year', 'genres', 'movieId']]
    response = random_recommendations.to_json(orient="index")

    return response


@app.route("/speech2text", methods=['POST'])
@cross_origin()
def speech2text():
    if not check_balance():
        return {"message": "Cost limit reached"}, 403

    if request.method != 'POST' or request.form['key'] != MIDDLEWARE_KEY:
        return {"message": "Forbidden request"}, 403

    if 'audiofile' not in request.files.keys():
        return {"message": "Audiofile missing!"}, 400

    if not request.files['audiofile'].filename.endswith('.wav'):
        return {"message": "Audiofile must be in .wav format!"}, 400

    # tmp store file
    save_path = str(datetime.now()) + "temp.wav"
    request.files['audiofile'].save(save_path)

    # continue processing...
    audio_config = speechsdk.audio.AudioConfig(filename=save_path)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config)

    # get text
    result = speech_recognizer.recognize_once_async().get()
    text = result.text

    print(text)

    # delete tmp file
    os.remove(save_path)

    # make request to azure albert
    r = requests.post(getEmbeddingUrl, json={'text': text})

    print(r)
    data = r.json()

    # transform response
    embedding_flattened, shape = data['embedding'], data['shape']
    embedding = torch.tensor(embedding_flattened).reshape(shape)

    # calculate cosine similarity between descriptions
    distances = pairwise_distances(
        descriptions, embedding.reshape(1, 768), metric='cosine')
    indices = np.argsort(distances, axis=0).tolist()
    indices = indices[:10]

    # get recommendations
    recommendations = pd.DataFrame(columns=df.columns)
    for index in indices:
        recommendations = pd.concat((recommendations, df.iloc[index, :]))

    # turn dummy encoding back
    genres = []

    for i in range(recommendations.shape[0]):
        df_tmp = recommendations.iloc[i, :].T[['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action',
                                               'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'IMAX', 'Documentary', 'War', 'Musical', 'Western', 'Film-Noir']]
        genres.append(list(df_tmp[df_tmp == 1].index))

    genres = pd.Series(genres)
    genres.index = recommendations.index
    recommendations['genres'] = genres

    # filter columns
    recommendations = recommendations[['title', 'description', 'poster_path',
                                       'vote_average', 'actor1', 'actor2', 'actor3', 'year', 'genres']]
    recommendations = recommendations.to_json(orient="index")
    return recommendations


@app.route("/testSpeech2Text", methods=['GET'])
@cross_origin()
def testSpeech2Text():
    if not check_balance():
        return {"message": "Cost limit reached"}, 403

    # send audio to speech2text
    audio_config = speechsdk.audio.AudioConfig(filename="test.wav")
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config)

    # get text
    result = speech_recognizer.recognize_once_async().get()
    text = result.text
    return text


# start application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5501)
