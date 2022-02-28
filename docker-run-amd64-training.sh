#! /usr/env /bin/bash
# docker run --rm --platform amd64 -it amd64-deploy bash
docker run \
    --rm \
    -v ./data_ignore:/app/model_save
    -it \
    amd64-deploy \
    bash
