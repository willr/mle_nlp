#! /usr/env /bin/bash
docker build --progress=plain  --platform amd64 -t amd64-training -f dockerfiles/Dockerfile-amd64-training .