import tensorflow as tf
import transformers
from transformers import AutoTokenizer
from src.constants import *
import os
import shutil
from huggingface_hub import HfApi


class ModelLoader:
    def __init__(self):
        model_path = os.path.join(ROOT_DIR, MODEL_DIR_NAME)
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)

        self.Model = tf.keras.models.load_model(MODEL_PATH)
        self.Tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)