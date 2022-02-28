#! /usr/env /bin/bash
if [ ! -f ./data_ignore/questions.csv.zip ]
then
	cp ./data/questions.csv.zip ./data_ignore/
fi
docker build --progress=plain  --platform amd64 -t nvidia-training -f dockerfiles/Dockerfile-nvidia-training .