import os
import onnxruntime

from transformers import AutoTokenizer

def start_session(model_dir, model_name):
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2-cedr-emotion-detection")
    session = onnxruntime.InferenceSession(os.path.join(model_dir, model_name))
    return session, tokenizer