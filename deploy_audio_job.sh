#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-cellular-cider-495602-r9}"
REGION="${REGION:-asia-east1}"
REPO="${REPO:-cloud-run-source-deploy}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/smart-audio:latest"
BUCKET_NAME="${BUCKET:-${PROJECT}-sheet-converter}"
SA_EMAIL="sheet-converter@${PROJECT}.iam.gserviceaccount.com"
JOB_NAME="smart-audio"

echo "=== Deploying Cloud Run Job: ${JOB_NAME} ==="
echo "Project:  ${PROJECT}"
echo "Region:   ${REGION}"
echo "Image:    ${IMAGE}"
echo "Bucket:   gs://${BUCKET_NAME}"
echo "SA:       ${SA_EMAIL}"

# 1. Upload ground truth fixtures to GCS as fallback
echo "Uploading fixtures to gs://${BUCKET_NAME}/tmp/smart_eval/fixtures/..."
gcloud storage cp fixtures/groundtruth/tinghai.json "gs://${BUCKET_NAME}/tmp/smart_eval/fixtures/tinghai.json" || true
gcloud storage cp fixtures/groundtruth/diaole.json "gs://${BUCKET_NAME}/tmp/smart_eval/fixtures/diaole.json" || true
gcloud storage cp fixtures/sections_truth.json "gs://${BUCKET_NAME}/tmp/smart_eval/fixtures/sections_truth.json" || true
gcloud storage cp fixtures/lead/* "gs://${BUCKET_NAME}/tmp/smart_eval/fixtures/lead/" || true

# 2. Build and push image via Cloud Build
echo "Building smart-audio container via Cloud Build (E2_HIGHCPU_8)..."
gcloud builds submit \
  --config=cloudbuild.audio.yaml \
  --project="${PROJECT}" \
  .

# 3. Deploy Cloud Run Job
echo "Deploying Cloud Run Job ${JOB_NAME}..."
gcloud run jobs deploy "${JOB_NAME}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --image="${IMAGE}" \
  --cpu=8 \
  --memory=32Gi \
  --task-timeout=3600 \
  --max-retries=0 \
  --service-account="${SA_EMAIL}" \
  --set-env-vars="BUCKET=${BUCKET_NAME},GOOGLE_CLOUD_PROJECT=${PROJECT}"

echo "=== Cloud Run Job ${JOB_NAME} successfully deployed! ==="
