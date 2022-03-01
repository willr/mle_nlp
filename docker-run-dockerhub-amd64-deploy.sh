#! /usr/env /bin/bash
# docker run --rm --platform amd64 -it amd64-deploy bash
docker pull docker.io/simplybez/mle_nlp:v1
docker run --rm -p 5000:5000 --env ML_MODEL_PATH='./data_ignore/bilstm5' --env TOKENIZER_JSON_PATH='./data_ignore/tokenizer.bilstm5.json' -it docker.io/simplybez/mle_nlp:v1

