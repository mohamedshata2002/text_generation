import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import functions as f
from fastapi import FastAPI

main = FastAPI()

# Load the tokenizer from file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model("./model")
max_len = 33


@main.post("/process-text/")
async def process_text_endpoint(text: str,words:int):
    processed_text = f.predict_words(text,model,words,max_len,tokenizer)

    return {"processed_text": processed_text}