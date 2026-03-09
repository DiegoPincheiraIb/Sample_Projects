# GCP ML API project

This project deploys the **Iris Classifier FastAPI** to **Google Cloud Run**. It demonstrates how to containerise a machine learning model and serve it as a REST API with minimal configuration.

## What You'll see in this project:
- Containerising an ML model with Docker
- Deploying to Google Cloud Run with one command
- Using environment variables for configuration
- Calling a live public API endpoint

## Prerequisites

1. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and authenticated
2. A GCP project with billing enabled
3. Cloud Run and Container Registry APIs enabled

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com containerregistry.googleapis.com
```

---

## Deploy with Cloud Run Source Deploy (simplest)

```bash
cd gcp_deployment/
gcloud run deploy iris-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi
```

Cloud Run will:
1. Build the Docker image via Cloud Build
2. Push it to Artifact Registry
3. Deploy and return a public HTTPS URL

---

## Deploy via Docker + Container Registry (manual)

```bash
# Build
docker build -t gcr.io/YOUR_PROJECT_ID/iris-api .

# Push
docker push gcr.io/YOUR_PROJECT_ID/iris-api

# Deploy
gcloud run deploy iris-api \
  --image gcr.io/YOUR_PROJECT_ID/iris-api \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated
```

---

## Test the Live API

```bash
SERVICE_URL=$(gcloud run services describe iris-api --region us-central1 --format "value(status.url)")

curl "$SERVICE_URL/health"

curl -X POST "$SERVICE_URL/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
This should return a prediction for the Iris species based on the input features.