import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np 

def process_seq(text,tokenizer,max_len):
    token_text = tokenizer.texts_to_sequences([text])[0]
    processed_text = pad_sequences([token_text],padding= "pre",maxlen=max_len)
    return processed_text

def predict_words(text,model,words,max_len,tokenizer):
    for _ in range(words):
        processed_text = process_seq(text,tokenizer,max_len)
        predictions = model.predict(processed_text,verbose=0)
        choice = np.random.choice([1,2,3])
        pred = np.argsort(predictions)[0][-choice]
        if pred != 0 :
             output_word  = tokenizer.index_word[pred]
        text +=" "+output_word
    return text
        


