#! /usr/env /bin/bash

# url where GLOVE vectors are located
# https://nlp.stanford.edu/data/glove.42B.300d.zip

curl http://downloads.cs.stanford.edu/nlp/data/glove.42B.300d.zip -o ./data_ignore/glove.42B.300d.zip
cd ./data_ignore
unzip ./glove.42B.300d.zip
