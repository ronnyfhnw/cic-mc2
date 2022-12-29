# flask imports
from flask import Flask, request, jsonify, render_template, redirect, url_for, make_response
from flask_cors import CORS, cross_origin

# general imports
import json
import logging
from datetime import datetime
import numpy as np
import pandas as pd

# imports albert
import torch
from transformers import AlbertConfig, AlbertModel, AlbertTokenizer

## static functions
def init():
    '''
    This function initialises the albert tokenizer and model.
    '''
    global albert, tokenizer
    albert = AlbertModel.from_pretrained('albert_model', output_hidden_states=True)
    tokenizer = AlbertTokenizer.from_pretrained('albert_tokenizer')
    albert.eval()

def preprocess(text:str):
    '''
    This function preprocesses text data from the speech2text (https://azure.microsoft.com/en-us/products/cognitive-services/speech-to-text/#features) and uses the tokenizer stored in the global variable.

    Params
    ------
        text:str - Sentencte from speech2text

    Returns
    -------
        token_tensor:torch.Tensor - Tensor with ids
        segments_tensor:torch.Tensor - Tensor with segments
    '''
    tokens = tokenizer.tokenize(text)
    token_tensor = torch.tensor(tokenizer.encode(tokens)).reshape(1,-1)
    segments_tensor = torch.tensor([1] * len(token_tensor)).reshape(1,-1)
    return token_tensor, segments_tensor


app = Flask(__name__)
init()

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Albert works fine</h1>'''

@app.route('/getEmbedding', methods=['POST'])
def getEmbedding():
    '''
    Transforms tokens_tensor and segments_tensor into Embedding with the Albert Model.

    Params
    ------
        token_tensor:torch.Tensor - Tensor with ids
        segments_tensor:torch.Tensor - Tensor with segments

    Returns
    -------
        embedding_vector:torch.Tensor - Vector with Embeddings from Albert
    '''
    if request.method != 'POST':
        return "Faulty request method!"

    # read and log input
    logging.info("Request received")
    data = request.json
    logging.info(input)

    # preprocess
    logging.info("Preprocessing ...")
    token_tensor, segments_tensor = preprocess(text=data['text'])

    # process
    logging.info("Processing ...")
    albert.eval()
    with torch.no_grad():
        output = albert(token_tensor, segments_tensor)
    hidden_states = output[2][1:]
    # hidden_states = output[2][-1]
        
    embedding = torch.stack(hidden_states, dim=0).mean(dim=0).mean(dim=1)
    # embedding = torch.mean(hidden_states, 1)
    logging.info("Processed:\n", embedding)

    # create json
    logging.info("Creating json ...")
    shape = list(embedding.shape)
    flattened_embedding = embedding.flatten().tolist()

    # check if tensor reproducable
    torch.eq(torch.tensor(flattened_embedding).reshape(shape), embedding)

    # build output
    output = {
        'shape': shape,
        'embedding': flattened_embedding
    }

    return jsonify(output)

# start application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5500)