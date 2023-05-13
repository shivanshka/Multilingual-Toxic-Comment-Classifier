import tensorflow as tf
import transformers
from transformers import AutoTokenizer
from src.constants import *


class ModelLoader:
    def __init__(self):
        self.Model = tf.keras.models.load_model(MODEL_PATH)
        self.Tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)