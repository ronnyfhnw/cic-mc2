from transformers import AlbertConfig, AlbertModel, AlbertTokenizer
from transformers.onnx import FeaturesManager
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
import json
import onnxruntime

def init():
    global tokenizer, albert, session
    # load ALBERT model
    albert = AlbertModel.from_pretrained('albert-base-v2', output_hidden_states=True).to(device)
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    albert.eval()

    # create onnx runtime and load onnx model
    # session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])

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
    token_tensor = torch.tensor(tokenizer.encode(tokens)).unsqueeze(0)
    segments_tensor = torch.tensor([1] * token_tensor.shape[1]).unsqueeze(0)
    return token_tensor, segments_tensor

def run(input:str) -> torch.Tensor:
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
    # read and log input
    logging.info("Request received")
    input = json.loads(input)
    logging.info(input)

    # preprocess
    logging.info("Preprocessing ...")
    token_tensor, segments_tensor = preprocess(text=input['text'])

    # process
    logging.info("Processing ...")
    albert.eval()
    with torch.no_grad():
        output = albert(token_tensor, segments_tensor)
    hidden_states = output[2][1:]
    embedding = torch.stack(hidden_states, dim=0).mean(dim=0).mean(dim=1)
    logging.info("Processed:\n", embedding)

    # create json
    logging.info("Creating json ...")
    


    return embedding
