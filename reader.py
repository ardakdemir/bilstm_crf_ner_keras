import os
import pickle
import sys
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array
from numpy import asarray
from numpy import zeros
import numpy
class Reader():
    def __init__(self,config,train=True):
        self.train = train
        self.config = config


def save_data(data,pickle_name):
    f = open(pickle_name,"wb")
    pickle.dump(data,f)
    f.close()
def get_glove(pk_f,glo_f):
    w2v_dict = {}
    if pk_f not in os.listdir():## load or read the Glove Vectors
        w2v_dict = get_w2v(glo_f)
        save_data(w2v_dict,pk_f)
    else:
        print("Loading vector dictionary from {}".format(pk_f))
        w2v_dict = pickle.load(open( pk_f, "rb" ))
    return w2v_dict


def get_w2v(file_name):
    print("Generating embedding dictionary from {}".format(file_name))
    w2v_dict = {}
    f = open(file_name).readlines()
    for line in f:
        ls = line.split()
        word = ls[0]
        coefs = asarray(ls[1:], dtype='float32')
        w2v_dict[word] = coefs
    return w2v_dict

def get_sent_length(sent):
    for i,x in enumerate(sent):
        if x == "ENDPAD":
            return i
    return len(sent)
def categorical_pred_to_tags(pred,tag_list):
    preds = []
    print(pred.shape)
    for p in pred:
        for i,p_ in enumerate(p):
            if p_!=0:
                preds.append(tag_list[i])
                break
    return preds
def idx_to_word(ids,char_list):
    s = ""
    for i in ids:
        s=s+char_list[i]
    return s
def get_raw_sents(filename):
    f = open(filename).readlines()
    sents = []
    tags = []
    sent = []
    tags_ = []
    tag_set = set()
    max_len = 0
    for line in f[1:]:
        if line == "\n":
            continue
        if "SAMPLE_START" in line or "[SEP]" in line:
            sents.append(sent)
            #tags.append(tags_)
            if len(sent)> max_len:
                max_len = len(sent)
            sent = []
            #tags_ = []
        else:
            ls = line.split()
            sent.append(ls[0])
            #tags_.append(ls[-1])
            #tag_set.add(ls[-1])
    if len(sent) > 0:
        sents.append(sent)
        #tags.append(tags_)
    return sents,max_len
def get_sents(filename):
    f = open(filename).readlines()
    sents = []
    tags = []
    sent = []
    tags_ = []
    tag_set = set()
    max_len = 0
    sent_lengths = []
    word_lengths = []
    sent_word_lengths = []
    max_word_length = 0
    for line in f[1:]:
        if line == "\n":
            continue
        if "SAMPLE_START" in line or "[SEP]" in line:
            sents.append(sent)
            sent_lengths.append(len(sent))
            tags.append(tags_)
            word_lengths.append(sent_word_lengths)
            if len(sent)> max_len:
                max_len = len(sent)
            sent = []
            sent_word_lengths = []
            tags_ = []
        else:
            ls = line.split()
            if len(ls[0])> max_word_length:
                max_word_length = len(ls[0])
            sent.append(ls[0])
            sent_word_lengths.append(len(ls[0]))
            tags_.append(ls[-1])
            tag_set.add(ls[-1])
    if len(sent) > 0:
        sents.append(sent)
        tags.append(tags_)
    return sents,tags,tag_set,max_len ,sent_lengths, max_word_length, word_lengths
def generate_vocab(vocab_file,sents = None,train=True):
    if vocab_file not in os.listdir() and train:## load or read the Glove Vectors
        vocab = set()
        for sent in sents:
            for word in sent:
                vocab.add(word)
        vocab = list(vocab)
        vocab2 = []
        for v in vocab:
            vocab2.append(v)
        vocab2.append("UNK")
        vocab2.append("ENDPAD")
        print("Saving vocab to {}".format(vocab_file))
        save_data(vocab2,vocab_file)
    else:
        try:
            print("Loading vocab from {}".format(vocab_file))
            vocab2 = pickle.load(open( vocab_file, "rb" ))
        except:
            raise Exception("Could not find vocab file in {}".format(vocab_file))
    word2idx = {w: i + 1 for i, w in enumerate(vocab2)}
    #tag2idx = {t: i for i, t in enumerate(train_tag_set)}

    return vocab2, word2idx
def get_embedding_matrix(w2v_dict,vocab,embed_dim = 300):
    embedding_matrix = zeros((len(vocab)+1, embed_dim))
    for i,word in enumerate(vocab):
        vec = w2v_dict.get(word)
        if vec is not None:
            embedding_matrix[i+1] = vec
    return embedding_matrix
def char_level_padding(X_chars,max_sent_length,max_word_length,pad_idx):
    X_chars_padded = []
    for sent in X_chars:
        X_sent = []
        for i in range(max_sent_length):
            X_word = []
            for j in range(max_word_length):
                try:
                    X_word.append(sent[i][j])
                except:
                    X_word.append(pad_idx)
            X_sent.append(X_word)
        X_chars_padded.append(X_sent)
    return X_chars_padded
def load_training_data(config):

    ## load embeddings
    pickle_name = config.pickle_name
    w2v_file = config.w2v_file
    w2v_dict = get_glove(pickle_name,w2v_file)

    ## get sentences
    train_file = config.train_file
    train_sents,train_tags,train_tag_set, train_max_len, train_sent_lengths, train_max_word_length , train_word_lengths= get_sents(train_file)
    if os.path.isfile(config.dev_file):
        dev_sents,dev_tags,dev_tag_set, dev_max_len , dev_sent_lengths, dev_max_word_length, dev_word_lengths = get_sents(config.dev_file)
        #dev_sents2 = get_sents2(dev_sents)
        train_max_len = max(train_max_len,dev_max_len)
        train_max_word_length = max(train_max_word_length,dev_max_word_length)
        for sent,leng,tag,w_leng in zip(dev_sents,dev_sent_lengths,dev_tags,dev_word_lengths):
            train_sent_lengths.append(leng)
            train_sent_lengths.append(w_leng)
            train_sents.append(sent)
            train_tags.append(tag)
        train_tag_set.union(dev_tag_set)
    #train_sent_lengths = [len(sent) for sent in train_sents]
    train_tag_list = list(train_tag_set)

    if config.tag_list_file not in os.listdir():
        print("Saving tag list to : {}".format(config.tag_list_file))
        save_data(train_tag_list,config.tag_list_file)

    vocab , word2idx = generate_vocab(config.vocab_file, sents=train_sents)
    tag2idx = {t: i for i, t in enumerate(train_tag_list)}

    unk_idx = len(vocab)-1 # id for UNK words
    unk_char_idx = len(config.char2idx)-1
    ## data
    X = [[word2idx.get(x,unk_idx) for x in sent] for sent in train_sents]
    X_chars = [[[config.char2idx.get(char,unk_char_idx) for char in word] for word in sent] for sent in train_sents]
    Y = [[tag2idx[x]  for x in tag] for tag in train_tags]
    print("Word before padding  ")
    print(X_chars[0][0])
    X = pad_sequences(maxlen=train_max_len, sequences=X, padding="post", value=len(vocab))
    Y = pad_sequences(maxlen=train_max_len, sequences=Y, padding="post", value=tag2idx["O"])
    X_chars = char_level_padding(X_chars,train_max_len,train_max_word_length,config.char2idx["PAD"])
    print(X_chars[0][0])
    print("X_chars shape: {}".format(numpy.array(X_chars[1]).shape) )
    print(idx_to_word(X_chars[0][0],config.chars))
    print(train_sents[0][0])
    assert(idx_to_word(X_chars[0][0][:train_word_lengths[0][0]],config.chars)==train_sents[0][0]), "Char ids contain errors"
    Y = [to_categorical(i, num_classes=len(train_tag_set)) for i in Y]


    ##embedding matrix
    embedding_matrix = get_embedding_matrix(w2v_dict,vocab,embed_dim=config.embedding_dim)

    return X, X_chars, Y, train_max_len,vocab,train_sent_lengths, train_tag_list,word2idx,tag2idx,embedding_matrix,train_max_word_length, train_word_lengths

def char_idx_to_arr(sent_idx,word_lengths,char_list):
    sent= ""

    for word,leng in zip(sent_idx,word_lengths):
        w = ""
        for i in range(leng):
            w+=char_list[word[i]]
        sent+=w
        sent+=" "
    return sent
def load_test_data(config):
    ## load embeddings
    pickle_name = config.pickle_name
    w2v_file = config.w2v_file
    w2v_dict = get_glove(pickle_name,w2v_file)
    test_file = config.test_file
    test_sents,test_tags,test_tag_set, test_max_len,test_sent_lengths,test_max_word ,test_word_lengths= get_sents(test_file)
    #test_sent_lengths = [len(sent) for sent in test_sents]
    chars = config.chars
    ## get session variables
    all_data_file = config.session_file
    print("Loading all data to {}".format(all_data_file))
    try:
        all_data_dict = pickle.load(open( all_data_file, "rb" ))
    except:
        raise Exception("Could not find all data file in {}".format(all_data_file))
    vocab = all_data_dict["vocab"]
    tag_list = all_data_dict["tag_list"]
    max_len = all_data_dict["max_length"]
    max_word_length = all_data_dict["max_word_length"]
    ##get indices
    word2idx = {w: i + 1 for i, w in enumerate(vocab)}
    tag2idx = {t: i for i, t in enumerate(tag_list)}

    unk_idx = len(vocab) -1
    unk_char_idx = len(config.char2idx)-1

    ## generate data
    X = [[word2idx.get(x,unk_idx) for x in sent] for sent in test_sents]
    Y = [[tag2idx[x]  for x in tag] for tag in test_tags]
    X_chars = [[[config.char2idx.get(char,unk_char_idx) for char in word] for word in sent] for sent in test_sents]

    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=len(vocab))
    Y = pad_sequences(maxlen=max_len, sequences=Y, padding="post", value=tag2idx["O"])
    X_chars = char_level_padding(X_chars,max_len,max_word_length,config.char2idx["PAD"])
    print("Printing characted ids for the first word")
    print(X_chars[0][0])

    Y = [to_categorical(i, num_classes=len(tag_list)) for i in Y]
    return X,X_chars,Y, max_len,vocab,test_sent_lengths,tag_list,word2idx,tag2idx,test_sents,max_word_length, test_word_lengths
