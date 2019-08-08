import os
import sys
import pickle
import string

class Config():

    def __init__(self,load=True):
        self.load = True
        self.w2v_file = "../../../Lab_sems/glove.6B.100d.txt"
        self.pickle_name = "../bilstmcrfner_dict.pk"
        self.train_file = "all_train.txt"
        self.dev_file = "all_dev.txt"
        self.test_file = "all_test.txt"
        self.test_raw = False
        self.use_chars = True
        self.embedding_dim = 100
        self.char_dim = 50
        self.char_lstm_units = 50
        self.lstm_hidden_units = 50
        self.lstm_layers = 3
        self.patience = 5
        self.weights_save_name = '../bilstm_crf_ner_weights.h5'
        self.dropout = 0.5
        self.batch_size = 64
        self.epochs = 50
        self.chars = [x for x in string.printable]
        self.char_pad = "PAD"
        self.char_unk = "UNKCHAR"
        self.chars.append(self.char_pad)
        self.chars.append(self.char_unk)
        self.char2idx = {self.chars[i] : i for i in range(len(self.chars))}
        self.validation_split = 0.1
        self.test_size = 0.1
        self.evaluate = True
        self.save_name = 'bilstm_crf_ner_weights.h5'
        self.test_raw = False
        self.vocab_file = "../vocab.pk"
        self.tag_list_file = "../tag_set.pk"
        self.session_file = "../all_data.pk"
        self.conll_file = "conll_out.txt"
    vocab = None
