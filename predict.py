from config import *
from bilstm_crf_model import *
from reader import *
from writer import *
import numpy as np
from keras.models import Model, Input

def get_intermediate_outputs(model,layer_name,input):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("char_embed").output)
    intermediate_output = intermediate_layer_model.predict(test)
    for sent,out in zip(test_sents,intermediate_output):
        for word, out_word in zip(sent,out):
            print("word")
            print(word)
            print("out")
            print(out_word)
def main():
    config = Config()
    X,X_chars,Y, max_len,vocab,test_sent_lengths,tag_list,word2idx,tag2idx, test_sents,max_word_length,test_word_lengths = load_test_data(config)
    config.vocab = vocab
    BiLSTM_CRF = BiLSTMCRF(config,max_len,max_word_length,len(tag_list),embed_flag = False)
    BiLSTM_CRF.build()
    model = BiLSTM_CRF.model
    print("Loading model from : {}".format(config.weights_save_name))
    model.load_weights(config.weights_save_name)
    crf = BiLSTM_CRF.crf
    model.summary()
    print("char_inds:")
    print(X_chars[0])
    print("Original sentence")
    print(arr_to_str(test_sents[0]))
    print("Retrieved sentence")
    print(char_idx_to_arr(X_chars[0],test_word_lengths[0],config.chars))
    if config.use_chars:
        preds = model.predict([X,np.array(X_chars).reshape(len(X_chars),max_len,max_word_length)])
    else:
        preds = model.predict(X)

    sents = [test_sent[:i] for i,test_sent in zip(test_sent_lengths,test_sents)]
    pred_arr = [categorical_pred_to_tags(pred[:i],tag_list) for i,pred in zip(test_sent_lengths,preds)]
    truth_arr = [categorical_pred_to_tags(truth[:i],tag_list) for i,truth in zip(test_sent_lengths,Y)]
    conll_writer(sents,truth_arr,pred_arr,config.conll_file)
if __name__ == "__main__":
    main()
