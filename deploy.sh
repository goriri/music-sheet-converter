#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-cellular-cider-495602-r9}"
REGION="${REGION:-asia-east1}"
SERVICE="${SERVICE:-sheet-converter}"
BUCKET_NAME="${BUCKET:-${PROJECT}-sheet-converter}"
SA_NAME="sheet-converter"
SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"

echo "=== Deploying ${SERVICE} to Cloud Run ==="
echo "Project:  ${PROJECT}"
echo "Region:   ${REGION}"
echo "Bucket:   gs://${BUCKET_NAME}"
echo "SA:       ${SA_EMAIL}"

# 1. Create storage bucket if missing
echo "Checking bucket gs://${BUCKET_NAME}..."
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" &>/dev/null; then
  echo "Creating bucket gs://${BUCKET_NAME} with uniform bucket-level access..."
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT}" \
    --location="${REGION}" \
    --uniform-bucket-level-access
else
  echo "Bucket gs://${BUCKET_NAME} already exists."
fi

# Configure lifecycle rule: delete after 30 days
LIFECYCLE_TMP="$(mktemp)"
cat > "${LIFECYCLE_TMP}" << 'EOF'
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 30}
    }
  ]
}
EOF
echo "Updating bucket lifecycle rule (delete after 30 days)..."
gcloud storage buckets update "gs://${BUCKET_NAME}" --lifecycle-file="${LIFECYCLE_TMP}"
rm -f "${LIFECYCLE_TMP}"

# 2. Create Service Account if missing
echo "Checking service account ${SA_EMAIL}..."
if ! gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT}" &>/dev/null; then
  echo "Creating service account ${SA_NAME}..."
  gcloud iam service-accounts create "${SA_NAME}" \
    --project="${PROJECT}" \
    --display-name="Sheet Converter Service Account"
else
  echo "Service account ${SA_NAME} already exists."
fi

# 3. Grant IAM permissions
echo "Granting roles/aiplatform.user on project..."
gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/aiplatform.user" \
  --condition=None \
  --quiet

echo "Granting roles/storage.objectAdmin on bucket..."
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin" \
  --quiet

# 4. Prepare environment variables
ENV_VARS="BUCKET=${BUCKET_NAME},GOOGLE_CLOUD_PROJECT=${PROJECT}"
if [ -n "${OMR_MODEL:-}" ]; then
  ENV_VARS="${ENV_VARS},OMR_MODEL=${OMR_MODEL}"
fi
QA_ARBITER_MODEL="${QA_ARBITER_MODEL:-claude-opus-5-5}"
QA_ARBITER_REGION="${QA_ARBITER_REGION:-global}"
QA_LLM_REVIEW="${QA_LLM_REVIEW:-0}"
ENV_VARS="${ENV_VARS},QA_ARBITER_MODEL=${QA_ARBITER_MODEL},QA_ARBITER_REGION=${QA_ARBITER_REGION},QA_LLM_REVIEW=${QA_LLM_REVIEW}"

# 5. Deploy to Cloud Run
echo "Deploying service ${SERVICE} via Cloud Build source deploy..."
gcloud run deploy "${SERVICE}" \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --source="." \
  --service-account="${SA_EMAIL}" \
  --no-cpu-throttling \
  --timeout=900 \
  --cpu=2 \
  --memory=2Gi \
  --min-instances=0 \
  --max-instances=3 \
  --concurrency=10 \
  --allow-unauthenticated \
  --quiet \
  --set-env-vars="${ENV_VARS}"

echo "Deployment complete."
