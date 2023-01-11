#!/usr/bin/env bash
docker build -t middleware .

az login
az acr login --name d0fc41ec9cbe4bdebd01d7b30df6209b

docker tag middleware d0fc41ec9cbe4bdebd01d7b30df6209b.azurecr.io/middleware
docker push d0fc41ec9cbe4bdebd01d7b30df6209b.azurecr.io/middleware

az acr build --registry d0fc41ec9cbe4bdebd01d7b30df6209b --image middleware:latest .   
kubectl apply -f ./deployment.yml
kubectl apply -f ./loadbalancer.yml