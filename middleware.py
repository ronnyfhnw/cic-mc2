# flask imports
from flask import Flask, request, jsonify, render_template, redirect, url_for, make_response
from flask_cors import CORS, cross_origin

# general imports
import json
from datetime import datetime
import numpy as np
import pandas as pd
from RecommenderSystem import *
from sklearn.metrics.pairwise import pairwise_distances

# azure imports
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice
import torch
from azureml.core import Workspace, Environment, conda_dependencies

# load env
with open('secrets.json', 'r') as f:
    secrets = json.load(f)

# load workspace
try:
    ws = Workspace(subscription_id=secrets['subscription_id'],
                    resource_group=secrets['resource_group'],
                    workspace_name=secrets['workspace_name'])
    # load aci webservice
    aci_service = AciWebservice(ws, name=secrets['aci_service_name'])
except:
    ws = None
    aci_service = None

# start app
app = Flask(__name__)

# load data for recommender
description_matrix = torch.load("data/pca_albert_title.pt", map_location=torch.device("cpu"))
title_matrix = torch.load("data/pca_albert_title.pt", map_location=torch.device("cpu"))
cast_matrix = torch.load("data/pca_albert_title.pt", map_location=torch.device("cpu"))
df = pd.read_csv("data/data_tmbd_cleaned.csv", delimiter=";", lineterminator="\n")

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
    data = request.get_json()
    text = data['text']
    assert type(text) == str
    
    # get embedding
    response = aci_service.run(input_data=text)
    assert type(response) == torch.Tensor

    # calculate cosine similarity between descriptions
    distances = pairwise_distances(embeddings, response.reshape(1, -1), metric='cosine').reshape(-1)
    indices = np.argsort(distances)
    indices = indices[1:11]
    distances = distances[indices]
    distances = 1 - distances

    # get recommendations
    recommendations = pd.DataFrame(columns=movie_info.columns)
    for i, index in enumerate(indices):
        recommendations = pd.concat((recommendations, movie_info.iloc[i,:]))

    # recommendations to json
    recommendations = recommendations.to_json()

    return recommendations

@app.route("/getRecommendationsByIds", methods=['POST'])
def getRecommendationsByIds():
    # get data
    ids = request.form.get("ids")
    ids = json.loads(ids)['ids']
    
    

    return jsonify(ids)


# start application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5500, debug=True)