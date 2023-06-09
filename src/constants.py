import os

ROOT_DIR = os.getcwd()
MODEL_DIR_NAME = "serving_model"
MODEL_NAME = "roberta-fine-tuned-2"
MODEL_PATH = os.path.join(ROOT_DIR, MODEL_DIR_NAME, MODEL_NAME)
TOKENIZER_FILE_NAME = "tokenizer"
TOKENIZER_PATH = os.path.join(ROOT_DIR, MODEL_DIR_NAME, TOKENIZER_FILE_NAME)
MAX_LEN =192
BUFFER_SIZE=2048