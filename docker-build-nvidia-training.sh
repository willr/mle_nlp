#! /usr/env /bin/bash
docker build --progress=plain  --platform amd64 -t nvidia-training -f dockerfiles/Dockerfile-nvidia-training .