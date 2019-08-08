from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras.layers import Input, Bidirectional, LSTM, Embedding, Dense, Dropout, Lambda, Concatenate
from keras_contrib.layers import CRF
import numpy as np
import keras.backend as K

from keras.models import load_model
import pickle
import string
import os
import sys

class BiLSTMCRF():

    def __init__(self,config,train_max_len,train_max_word,tag_size,embedding_matrix=None,embed_flag = True):
        self.config = config
        self.train_max_len = train_max_len
        self.embedding_matrix = embedding_matrix
        self.max_word_length =  train_max_word
        self.tag_size = tag_size
        self.embed_flag = embed_flag
        self.chars = config.chars
        self.nchars = len(self.chars)
    def build(self):
        inputs = []
        print("Maximum lengths:")
        print(self.train_max_len)
        input = Input(shape=(self.train_max_len,))
        inputs.append(input)
        input_chars = Input(shape=(self.train_max_len,self.max_word_length))
        if self.embed_flag:
            word_embeddings = Embedding(input_dim=len(self.embedding_matrix), output_dim=self.config.embedding_dim,weights = [self.embedding_matrix],
                          input_length=self.train_max_len, mask_zero=True,trainable=True,name="word_embed")(input)  # 300-dim embedding
        else:
            word_embeddings = Embedding(input_dim=len(self.config.vocab)+1, output_dim=self.config.embedding_dim,
                          input_length=self.train_max_len, mask_zero=True,trainable=True,name="word_embed")(input)
        if self.config.use_chars :
            inputs.append(input_chars)
            char_embeds= TimeDistributed(Embedding(input_dim=self.nchars, output_dim=self.config.char_dim,input_length=(self.max_word_length),
                           mask_zero=True,trainable=True),name="char_embed")(input_chars)
            #print("Shape of char_embeds : {}".format(char_embeds.value.shape))
            s = K.shape(char_embeds)
            s2 = K.shape(word_embeddings)

            print("Word embedding shape: {}".format(s2))
            #print("Char embedding shape: {} {} {}".format(s[0],s[1],s[2]))
            #char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], self.config.char_dim)))(char_embeds)
            #print("After char embedding shape: {}".format(K.shape(char_embeddings)))
            #fwd_state = LSTM(self.config.lstm_hidden_units, return_state=True, name='fw_char_lstm')(char_embeddings)[-2]
            #bwd_state = LSTM(self.config.lstm_hidden_units, return_state=True, go_backwards=True, name='bw_char_lstm')(char_embeddings)[-2]
            #char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
            char_embeddings = TimeDistributed(Bidirectional(LSTM(units=self.config.char_lstm_units, return_sequences=False,
                                   recurrent_dropout=self.config.dropout)))(char_embeds)

            #char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * self.config.lstm_hidden_units]))(char_embeddings)

            word_embeddings = Concatenate(axis=-1)([word_embeddings, char_embeddings])

        word_embeddings = Dropout(self.config.dropout)(word_embeddings)
        for i in range(self.config.lstm_layers):
            word_embeddings = Bidirectional(LSTM(units=self.config.lstm_hidden_units, return_sequences=True,
                                   recurrent_dropout=self.config.dropout),name="BiLSTM_{}".format(i))(word_embeddings)  # variational biLSTM
        model = TimeDistributed(Dense(50, activation="relu"))(word_embeddings)  # a dense layer as suggested by neuralNer
        crf = CRF(self.tag_size)  # CRF layer
        out = crf(model)  # output
        model = Model(inputs, out)
        self.crf = crf
        self.out = out
        self.model = model
