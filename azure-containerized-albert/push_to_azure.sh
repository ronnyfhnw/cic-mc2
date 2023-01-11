#!/usr/bin/env bash

az login
az acr login --name d0fc41ec9cbe4bdebd01d7b30df6209b
docker tag mock d0fc41ec9cbe4bdebd01d7b30df6209b.azurecr.io/
docker push d0fc41ec9cbe4bdebd01d7b30df6209b.azurecr.io/azure-containerized-albert