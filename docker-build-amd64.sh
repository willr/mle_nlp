#! /usr/env /bin/bash
docker build --progress=plain  --platform amd64 -t amd64-deploy -f dockerfiles/Dockerfile-amd64-deploy .