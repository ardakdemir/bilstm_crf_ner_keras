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
import numpy as np
from keras.models import load_model
import pickle
import os
import sys


def get_w2v(file_name):
    w2v_dict = {}
    f = open(file_name).readlines()
    for line in f:
        ls = line.split()
        word = ls[0]
        coefs = asarray(ls[1:], dtype='float32')
        w2v_dict[word] = coefs
    return w2v_dict



def get_sents(filename):
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
            tags.append(tags_)
            if len(sent)> max_len:
                max_len = len(sent)
            sent = []
            tags_ = []
        else:
            ls = line.split()
            sent.append(ls[0])
            tags_.append(ls[-1])
            tag_set.add(ls[-1])
    if len(sent) > 0:
        sents.append(sent)
        tags.append(tags_)
    return sents,tags,tag_set,max_len


def idx_to_arr(idx_arr,vocab):
    arr = []
    for idx in idx_arr:
        arr.append(vocab[idx-1])
    return arr
def arr_to_str(arr):
    return " ".join([x for x in arr])

def categorical_pred_to_tags(pred,tag_list):
    preds = []
    print(pred.shape)
    for p in pred:
        for i,p_ in enumerate(p):
            if p_!=0:
                preds.append(tag_list[i])
                break
    return preds
def get_sent_length(sent):
    for i,x in enumerate(sent):
        if x == "ENDPAD":
            return i
    return len(sent)
get_sents2 = lambda sents : [arr_to_str(sent) for sent in sents]

if pickle_name not in os.listdir():
    w2v_dict = get_w2v(w2v_file)
    f = open(pickle_name,"wb")
    pickle.dump(w2v_dict,f)
    f.close()
else:
    w2v_dict = pickle.load(open( pickle_name, "rb" ))

train_file = "train.txt"
dev_file = "dev.txt"
test_file = "test.txt"
train_sents,train_tags,train_tag_set, train_max_len = get_sents(train_file)
dev_sents,dev_tags,dev_tag_set, dev_max_len = get_sents(dev_file)

#dev_sents2 = get_sents2(dev_sents)
train_max_len = max(train_max_len,dev_max_len)
for sent,tag in zip(dev_sents,dev_tags):
    train_sents.append(sent)
    train_tags.append(tag)

train_sent_length = [len(sent) for sent in train_sents]
train_tag_set.union(dev_tag_set)
train_tag_list = list(train_tag_set)

vocab = set()
for sent in train_sents:
    for word in sent:
        vocab.add(word)
vocab = list(vocab)
vocab2 = []
for v in vocab:
    vocab2.append(v)
vocab2.append("UNK")
vocab2.append("ENDPAD")

word2idx = {w: i + 1 for i, w in enumerate(vocab2)}
tag2idx = {t: i for i, t in enumerate(train_tag_set)}

X = [[word2idx[x] for x in sent] for sent in train_sents]
Y = [[tag2idx[x]  for x in tag] for tag in train_tags]

X = pad_sequences(maxlen=train_max_len, sequences=X, padding="post", value=len(vocab2))
Y = pad_sequences(maxlen=train_max_len, sequences=Y, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=len(train_tag_set)) for i in Y]

print("Original sentence : ")
print("{}".format(arr_to_str(train_sents[0])))
print("Encoded sentence : ")
print("{}".format(arr_to_str(idx_to_arr(X[0],vocab2))))
embedding_matrix = zeros((len(vocab2)+1, 300))
for i,word in enumerate(vocab2):
    vec = w2v_dict.get(word)
    if vec is not None:
        embedding_matrix[i+1] = vec

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

input = Input(shape=(train_max_len,))
model = Embedding(input_dim=len(vocab2)+1, output_dim=300,weights = [embedding_matrix],
                  input_length=train_max_len, mask_zero=True,trainable=False)(input)  # 300-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(len(train_tag_set))  # CRF layer
out = crf(model)  # output
model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()
history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=20,
                    validation_split=0.1, verbose=1)
save_name = 'bilstm_crf_ner_weights.h5'
print("Saving model weights to : {}".format(save_name))
model.save_weights(save_name)
pred = model.predict(X_te)
print("Preds")
for i in range(10):
    sent_idx = X_te[i]
    word_arr = idx_to_arr(sent_idx,vocab2)
    #sent = idx_to_arr(sent_idx,vocab)
    print(arr_to_str(word_arr))
    sent_len = get_sent_length(word_arr)
    print(sent_len)
    truth = y_te[i][:sent_len]
    #print(truth[i])
    pred_arr = categorical_pred_to_tags(pred[i][:sent_len],train_tag_list)
    truth_arr = categorical_pred_to_tags(truth,train_tag_list)
    for w,p,t in zip(word_arr,pred_arr,truth_arr):
        print("{} {} {}".format(w,t,p))


print(train_tag_list)
#tag_arr  = idx_to_arr(pred[0],train_tag_list)
#print("Predictions for:\n {}".format(arr_to_str(word_arr)))
#print(tag_arr)
