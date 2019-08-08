import os
import sys


def idx_to_arr(idx_arr,vocab):
    arr = []
    for idx in idx_arr:
        arr.append(vocab[idx-1])
    return arr
def arr_to_str(arr):
    return " ".join([x for x in arr])

def categorical_pred_to_tags(pred,tag_list):
    preds = []
    #print(pred.shape)
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
def conll_writer(tokens,truths, preds,outname):
    out = open(outname,"w")
    assert len(tokens) == len(truths) == len(preds), "Lengths are not matched!!!"
    for sent,trut,pred in zip(tokens,truths,preds):
        for w,t,p in zip(sent,trut,pred):
            out.write("{}\t{}\t{}\n".format(w,t,p))
        out.write("\n")
    out.close()
