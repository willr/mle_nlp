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

## Running the tests

Dev-only dependencies (pytest, mlflow) live in `requirements-dev.txt`, layered
on top of whichever conda env you already activated (see above) rather than
hand-edited into the pinned `environments/*.yaml` snapshots:
- pip install -r requirements-dev.txt
- pytest

The suite lives in `tests/` and covers text preprocessing (`normalize_text`),
the evaluation metrics math (`src/evaluate.py`), and the Flask API layer. The
API tests mock the ML boundary (`ml_predict`) rather than loading a real
model, so they run fast and don't require a trained model file on disk;
actual model correctness is validated by the eval report instead (see below).

## Experiment tracking and the eval report

`python src/training_pipeline.py` now wraps the run in an MLflow run: params
from `constants.py`, best validation loss/accuracy, held-out test metrics
(accuracy/precision/recall/F1/ROC-AUC/confusion matrix), and artifacts (the
saved model dir, tokenizer json, eval report json) are all logged.
- View results: `mlflow ui` (from the repo root, after a training run)
- The eval report also lands on disk at
  `data_ignore/eval_report.<VERSION>.json` (`VERSION` is the constant in
  `constants.py`)

## v1 -> v2 rework log

This repo started as a working training/serving pipeline that had never been
scored against a held-out test set. The rework below adds the things a
production-minded MLE workflow expects: a real held-out evaluation, no
training/serving skew, experiment tracking, and test coverage. Gaps found
and fixed:

1. **A held-out split was prepared but never scored.** `training_pipeline.py`
   (the script actually used for training) defined `TEST_SIZE`/`RANDOM_STATE`
   in `constants.py` and imported `train_test_split`, but never called it -
   there was no test split at all in that script. The exploration notebook
   (`notebooks/capstone.ipynb`) did better and worse at once: it built a real
   train/test split and held-out labels, but never computed a single metric
   against them - model selection across several hand-tuned variants
   (different training-patience values) was done by eyeballing predictions
   on a handful of fixed example question pairs instead. The notebook's
   tokenizer was also fit on train **and** test text combined, leaking test
   vocabulary into the vocab index.
   Fixed: `training_pipeline.py` now does a real stratified train/test split
   and scores the model on the held-out set via `src/evaluate.py`
   (accuracy/precision/recall/F1/ROC-AUC/confusion matrix), logged through
   MLflow. The `Tokenizer` is fit on the training split only, to avoid
   repeating the notebook's leakage mistake.
2. **Preprocessing was hand-duplicated** between `training_pipeline.py` and
   `webapp/textsimilar/ml_process.py` - identical ~40-line regex block,
   copy-pasted. Fixed: both now import a single `normalize_text()` from
   `src/text_preprocessing.py`.
3. **`MAX_SEQUENCE_LENGTH` was also duplicated** (hardcoded again in
   `ml_process.py` instead of imported from `constants.py`). Fixed: same
   root cause as #2, same fix - `ml_process.py` now imports it from
   `constants.py`.
4. **Zero automated tests.** Fixed: a real pytest suite now lives in
   `tests/`.
5. **A live API bug**: `routes.py`'s `sm_to_json()` set `json['q2'] = sm.q1`
   (copy-paste), so the API always echoed `q1` back twice instead of
   returning `q2`. Fixed, with a regression test locking in the fix.
6. **`Environment.TEST` config loading was broken**: `create_app()` loaded
   the test config via a string import path to a module that doesn't exist,
   while the other two environments passed class objects directly - so the
   test Flask config path had never actually been exercised. Fixed.
7. **Serving is single-worker** (`gunicorn_config.py`) with the model as a
   lazy global singleton - functional but not horizontally scalable. Not
   yet addressed; planned alongside a move to FastAPI behind autoscaling.
8. **Prediction history is in-memory only** (`PastTS` global list) - no
   persistence, so no basis for drift/monitoring. Not yet addressed.
