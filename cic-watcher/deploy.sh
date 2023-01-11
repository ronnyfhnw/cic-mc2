#!/usr/bin/env bash
docker build -t cic-watcher .

az login
az acr login --name d0fc41ec9cbe4bdebd01d7b30df6209b

docker tag cic-watcher d0fc41ec9cbe4bdebd01d7b30df6209b.azurecr.io/cic-watcher
docker push d0fc41ec9cbe4bdebd01d7b30df6209b.azurecr.io/cic-watcher

az acr build --registry d0fc41ec9cbe4bdebd01d7b30df6209b --image cic-watcher:latest .   
kubectl apply -f ./deployment.yml
kubectl apply -f ./loadbalancer.yml