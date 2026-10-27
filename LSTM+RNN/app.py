import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load the LSTM model

model=load_model('next_word_lstm.h5')



# Load the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)


## Function to predict the next word
## Function to predict the next word using LSTM

def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):]

    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')

    predicted=model.predict(token_list,verbose=0)

    predicted_word_index=np.argmax(predicted,axis=1)[0]

    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None


# Load the GRU model
model2=load_model('next_word_gru.h5')


# load the tokenizer for GRU model
with open('tokenizer_gru.pickle','rb') as handle:
    tokenizer_gru=pickle.load(handle)


# Function to predict the next word using GRU
def predict_next_word_gru(model2,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):]

    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')

    predicted=model2.predict(token_list,verbose=0)

    predicted_word_index=np.argmax(predicted,axis=1)[0]

    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None


# Streamlit app

st.title("Next Word Prediction with LSTM and GRU")
input_text=st.text_input("Enter a text sequence:")
if st.button("Predict Next Word"):
    max_sequence_len=model.input_shape[1]+1

    # Predict the next word using LSTM
    next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Next Word Prediction (LSTM): {next_word}")


    # Predict the next word using GRU
    next_word_gru=predict_next_word_gru(model2,tokenizer_gru,input_text,max_sequence_len)
    st.write(f"Next Word Prediction (GRU): {next_word_gru}")