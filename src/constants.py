
MAX_SEQUENCE_LENGTH = 60  
MAX_NUM_WORDS = 200000  # There are about 201000 unique words in training dataset, 200000 is enough for tokenization
EMBEDDING_DIM = 300  # word-embedded-vector dimension(300 is for 'glove.42B.300d')
N_HIDDEN = 512
N_DENSE = 256

DROPOUT_RATE_LSTM = 0.10 # drop-out possibility, random set to avoid outfitting
DROPOUT_RATE_DENSE = 0.15

ACTIVE_FUNC = 'relu'
MODEL_TRAINING_PATIENCE = 5
VERSION = f'bilstm{MODEL_TRAINING_PATIENCE}-2'

PATH_TO_GLOVE_FILE = './data_ignore/glove.42B.300d.txt'
PATH_TO_QUESTIONS = './data/questions.csv.zip'

TEST_SIZE = 0.95
RANDOM_STATE = 42

