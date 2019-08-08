# bilstm_crf_ner_keras

BiLSTM CRF with character embeddings implemented in Keras


Very easy to use Deep Learning model for Named Entity Recognition and any other Sequence Labeling Task where the input is a
list of tokens represented in CONLL format.

Just update the variables in config.py file according to your needs (train_file, test_file etc) and run either train_model.py for 
training or predict.py for prediction. 

***Reminder!***: You must provide a working .h5 file containing the model weights in order to run the model in prediction mode.

## Training
```
python train_model.py
```

## Prediction

```
python predict.py
```
