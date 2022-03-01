# mle_nlp

The ml code in this repo is allows for trainging a Keras to identify duplicate questions.
The data for training the model to comes from Kaggle (https://www.kaggle.com/quora/question-pairs-dataset/version/2), specifically
the Quora Question Pairs Dataset (login required to download the dataset).  This dataset has approx 400k pairs of questions with a 
label of duplicate or not.  This data comes from the questions posted to the Quora site it self.

This repo contains:
- [notebooks/capstone.ipynb] the jupyter notebook where the data exploration task and model develop originally occurred.
- [src/*.py] python scripts to build, setup and train the Keras based model, with the question data downloaded from Kaggle.
- [environments/*.yaml] conda environment files to building the required python environment and dependencies on different platforms. (linux, apple)
- [dockerfiles/*] docker files for building the container in which to run the code to train the model (not required), and execute the model predictions for deployment
- [docker-build-*-training.sh] scripts to build a docker container to conduct training of the model (limited to CPU based [slow]).  GPU based training should 
    be conducted on the host, with access to the GPU and the proper conda environment installed.
- [docker-build-*-run.sh] scripts to build a docker container for deployment on the respective host.  This will host and serve the model 
    with a simple website to demonstrate the model in action.
