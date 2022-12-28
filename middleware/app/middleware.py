# flask imports
from flask import Flask, request, jsonify, render_template, redirect, url_for, make_response
from flask_cors import CORS, cross_origin

# general imports
import json
import requests
from datetime import datetime
import numpy as np
import pandas as pd
from RecommenderSystem import *
from sklearn.metrics.pairwise import pairwise_distances
import torch

getEmbeddingUrl = "http://127.0.0.1:5500/getEmbedding"

# start app
app = Flask(__name__)

# load data for recommender
descriptions = torch.load("data/avgpooled_albert_sentence_embeddings_float32.pt", map_location=torch.device("cpu")).detach().numpy()
description_matrix = np.array(torch.load("data/pca_albert_description.pt", map_location=torch.device("cpu")))
title_matrix = np.array(torch.load("data/pca_albert_title.pt", map_location=torch.device("cpu")))
cast_matrix = np.array(torch.load("data/pca_albert_cast.pt", map_location=torch.device("cpu")))
df = pd.read_csv("data/data_tmbd_cleaned.csv", delimiter=";", lineterminator="\n")

# filter matrices
description_matrix = description_matrix[list(df.index),:]
descriptions = descriptions[list(df.index),:]
title_matrix = title_matrix[list(df.index),:]
cast_matrix = cast_matrix[list(df.index),:]

print(descriptions)

# build recommender system
movie_info = ContentBasedRecommender.build_movie_info_matrix(df)
mapping_matrix = ContentBasedRecommender.build_mapping_matrix(df)

matrix_dict = {
        'movie_info_matrix': movie_info,
        'description_matrix': description_matrix,
        'title_matrix': title_matrix,
        'cast_info_matrix': cast_matrix
}

RS = ContentBasedRecommender(
            mapping_matrix=mapping_matrix,
            **matrix_dict,
            scaling_kwargs=None
        )

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/getRecommendationByText', methods=['POST'])
def getRecommendationByText():
    # get data
    data = request.json
    text = data['text']
    assert type(text) == str
    
    # make request to azure albert
    response = requests.post(getEmbeddingUrl, json={'text':text})
    data = response.json()

    # transform response
    embedding_flattened, shape = data['embedding'], data['shape']
    embedding = torch.tensor(embedding_flattened).reshape(shape)

    print(embedding.shape, descriptions.shape)
    
    # calculate cosine similarity between descriptions
    distances = pairwise_distances(descriptions, embedding.reshape(768), metric='cosine')
    indices = np.argsort(distances, axis=0).tolist()
    indices = indices[:10]
    print(indices)
    
    # get recommendations
    recommendations = pd.DataFrame(columns=df.columns)
    for index in indices:
        recommendations = pd.concat((recommendations, df.iloc[index, :]))

    # recommendations to json
    recommendations = recommendations.reset_index()
    recommendations = recommendations[['title']]
    recommendations = recommendations.to_json(orient="index")

    return recommendations

@app.route("/getRecommendationsByIds", methods=['POST'])
def getRecommendationsByIds():
    # get data 
    data = request.json
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

# start application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5501, debug=True)