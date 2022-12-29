#!/bin/bash

AKS_RESOURCE_GROUP=cic
AKS_CLUSTER_NAME=fhnw-cic-rsfm
ACR_RESOURCE_GROUP=cic
ACR_NAME=d0fc41ec9cbe4bdebd01d7b30df6209b

# Get the id of the service principal configured for AKS
CLIENT_ID=$(az aks show --resource-group $AKS_RESOURCE_GROUP --name $AKS_CLUSTER_NAME --query "servicePrincipalProfile.clientId" --output tsv)

# Get the ACR registry resource id
ACR_ID=$(az acr show --name $ACR_NAME --resource-group $ACR_RESOURCE_GROUP --query "id" --output tsv)

# Create role assignment
az role assignment create --assignee $CLIENT_ID --role acrpull --scope $ACR_ID