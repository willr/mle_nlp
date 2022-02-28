#! /usr/env /bin/bash
if [ ! -f ./data_ignore/questions.csv.zip ]
then
	cp ./data/questions.csv.zip ./data_ignore/
fi
docker build --progress=plain  --platform amd64 -t amd64-training -f dockerfiles/Dockerfile-amd64-training .