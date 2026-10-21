#!/usr/bin/env bash
export PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
export REGION="${REGION:-us-central1}"
export NAME="${NAME:-rfd4-proteina}"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${NAME}/api:sample"
export BUILD_SA="${BUILD_SA:-${NAME}-build@${PROJECT_ID}.iam.gserviceaccount.com}"
export NODE_SA="${NODE_SA:-${NAME}-node@${PROJECT_ID}.iam.gserviceaccount.com}"
export RFD4_COMMIT="${RFD4_COMMIT:-f3061df1166d51e7281fe7e637c6e6400fca4b83}"
export RFD4_REPO="${RFD4_REPO:-https://github.com/RosettaCommons/RFD4-Proteina.git}"
