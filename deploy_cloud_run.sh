#!/bin/bash
set -e

# Configuration
PROJECT_ID="bitcoin-prediction-4739108-b3" # REPLACE THIS
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/bitcoin-system"

echo "========================================================"
echo "Deploying Bitcoin Price Prediction System to Cloud Run"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "========================================================"

# 1. Build and Push Image to Container Registry
echo "[1/3] Building and Pushing Docker Image..."
# Note: You can also use Artifact Registry (us-central1-docker.pkg.dev/...)
gcloud builds submit --tag $IMAGE_NAME .

# 2. Deploy API Service
echo "[2/3] Deploying API Service..."
gcloud run deploy bitcoin-api \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars SERVICE_TYPE=api

# 3. Deploy Frontend Service
echo "[3/3] Deploying Frontend Service..."
gcloud run deploy bitcoin-frontend \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --set-env-vars SERVICE_TYPE=frontend

echo "========================================================"
echo "Deployment Complete!"
echo "Check the URLs above for your services."
echo "========================================================"
