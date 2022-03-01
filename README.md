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
    be conducted on the host, with access to the GPU and the proper conda environment installed. Replace the * with the correct host environment.
- [docker-run-*-training.sh] script to run previously built docker container to train the model.  This is CPU based training, nvidia docker was not yet     
    implmented. Training via GPU currently requires running directly on the host inside a configured conda environment on the host. Replace the * with the correct host environment.
- [docker-build-*-deploy.sh] scripts to build a docker container for deployment on the respective host.  This will host and serve the model 
    with a simple website to demonstrate the model in action. Replace the * with the correct host environment.
- [docker-run-*-deploy] script to run previously built docker container to deploy and serve the model, hosted in a flask env to enable access via simple website
    to demonstrate the model in action.  Replace the * with the correct host environment.


Instructions for deploying the conda training environment locally:
- Install conda [https://docs.anaconda.com/anaconda/install/]
- Create a local host based model training environment and activate it
    - conda env create -f environments/environment.tf_26_training_gpu.amd64.linux.yaml
    - conda activate tf_26_training_gpu
- Execute the training pipeline
    - python src/training_pipeline.py
    - when this completes the model will be saved into ./data_ignore
    - model name is the constant VERSION in the constants.py file

To serve the model you just created
- Create a local host based model deploy environment and activate it
    - conda env create -f environments/environment.tf_26_deploy_gpu.amd64.linux.yaml
    - conda activate tf_26_deploy_gpu
- Execute the model serving website
    - python src/web.app.py
    - deployment env defaults to Development, edit in web.app.py line 3
    - model to load controlled by src/webapp/textsimilar/config.py, edit the appropriate env you have configured
- either point a web browser at http://localhost:5000 or run: python src/json_submit.py to test the model serving
    
To serve the model via docker
- Build a docker container with a previous trained model
    - edit the model file by editing the file, [docker-build-amd64-deploy.sh] the two docker "build-arg" should be edited
        - trained model file
        - serialized tokenizer json file
- Execute docker build of the deploy container
    - bash docker-build-amd64-deploy.sh
- Execute the docker container you just built
    - bash docker-run-amd64-deploy.sh
- either point a web browser at http://localhost:5000 or run: python src/json_submit.py to test the model serving

To pull and run a prebuilt model deploy container (large 4GB+)
    - bash docker-run-dockerhub-amd64-deploy.sh

Externally the system is accessible at:
    - http://mle-nlp.grinful.com:5000/
    - Either use the web UI, or send a request via existing script:
        - aws_json_submit.py - send json request to AWS hosted instance via REST call
        - json_submit.py - localhost json request via REST call
