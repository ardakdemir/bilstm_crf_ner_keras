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
from keras_contrib.layers import CRF
from keras import callbacks,losses
import numpy as np
from keras.models import load_model
import pickle
import os
import sys
from reader import *
from bilstm_crf_model import *
from  config import *

def save_all_data(all_data_file,vocab,train_tag_list,train_max_len,train_max_word_length):
    print("Saving all data to {}".format(all_data_file))
    all_data_dict = {}
    all_data_dict["vocab"] = vocab
    all_data_dict["tag_list"] = train_tag_list
    all_data_dict["max_length"]  = train_max_len
    all_data_dict["max_word_length"]  = train_max_word_length
    f = open(all_data_file,"wb")
    pickle.dump(all_data_dict,f)
    f.close()

def train_model():
    config = Config()
    CB = callbacks.EarlyStopping(monitor="val_loss", patience=config.patience, mode="auto", restore_best_weights=True)
    mcp_save = callbacks.ModelCheckpoint(config.save_name, save_best_only=True, monitor='val_loss', mode='min')
    X,X_chars,Y, train_max_len,vocab,train_sent_lengths, train_tag_list,word2idx,tag2idx,embedding_matrix , train_max_word_length, train_word_lengths= load_training_data(config)
    Model = BiLSTMCRF(config,train_max_len,train_max_word_length,len(train_tag_list),embedding_matrix=embedding_matrix)
    Model.build()
    model = Model.model
    crf = Model.crf
    model.summary()
    all_data_file = config.session_file
    save_all_data(all_data_file,vocab,train_tag_list,train_max_len,train_max_word_length)
    s = np.array(X_chars).shape
    print("X_chars shape: {}".format(s))
    print("X shape: {}".format(np.array(X).shape))
    model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    if config.use_chars:
        history = model.fit([np.array(X),np.array(X_chars).reshape(len(X_chars),train_max_len,train_max_word_length)], np.array(Y), batch_size=config.batch_size, epochs=config.epochs, validation_split=config.validation_split, verbose=1,callbacks= [CB,mcp_save])
    else:
        history = model.fit(np.array(X), np.array(Y), batch_size=config.batch_size, epochs=config.epochs, validation_split=config.validation_split, verbose=1,callbacks= [CB,mcp_save])
    #print("Saving model weights to : {}".format(config.save_name))
    #model.save_weights(config.save_name)

if __name__=="__main__":
    train_model()
