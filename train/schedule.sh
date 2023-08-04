#!/bin/bash
set -e  # exit on error

export EPFML_STORE_S3_SECRET_KEY=NhvIp=MvCRN+ywScTPQ9Pr8K1=H+YWsEaklUbxt6
export EPFML_STORE_S3_ACCESS_KEY=3ZYKH95QSGUFLJR7NPJX
export EPFML_LDAP=vcanard
export EPFML_STORE_S3_BUCKET=13319-806957bd411dbc36a20db4f5c5f90f7b
export EPFML_STORE_S3_ENDPOINT=https://s3.epfl.ch


USER=vcanard
LAB=linx

WANDB_PROJECT="gpt2-parity"
WANDB_RUN_GROUP="test-try-emb-encoder-02"
WANDB_API_KEY=`python -c "import wandb; print(wandb.api.api_key)"`
CODE_BUNDLE=`epfml bundle pack .`

# Generate a unique ID for wandb. This makes sure that automatic restarts continue with the same job.
RUN_ID=`python -c "import wandb; print(wandb.util.generate_id())"`;
RUN_FILE="python train.py config/wikitext/wikitext.json save"


runai submit \
    --name ${WANDB_RUN_GROUP}-${RUN_ID} \
    --environment WANDB_PROJECT=$WANDB_PROJECT \
    --environment WANDB_RUN_GROUP=$WANDB_RUN_GROUP \
    --environment WANDB_RUN_ID=$RUN_ID \
    --environment WANDB_API_KEY=$WANDB_API_KEY \
    --gpu 1 \
    --node-type "G10" \
    --image ic-registry.epfl.ch/linx/bondasch-pytorch-base:latest \
    --large-shm \
    --host-ipc \
    --environment DATA_DIR=/home/$USER/data \
    --environment EPFML_LDAP=$USER \
    --command -- \
        /entrypoint.sh \
        su $USER -c \
        \"epfml bundle exec $CODE_BUNDLE -- $RUN_FILE\";
