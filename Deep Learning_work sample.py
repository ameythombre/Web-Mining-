

import matplotlib.pyplot as plt
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Input, Flatten, Concatenate
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.metrics import classification_report
from keras.models import Model
from keras.preprocessing.text import Tokenizer


def sentiment_cnn(text):
    
    # Assigning Values and changing them to find the accuracy
    MAX_NB_WORDS=10000
    MAX_DOC_LEN=500
    EMBEDDING_DIM=1000
    BATCH_SIZE = 16
    NUM_EPOCHES = 20
    
    # Tokenizing and adding grams  
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(text["text"])
    sequences = tokenizer.texts_to_sequences(text["text"])
    padded_sequences = pad_sequences(sequences,                                      maxlen=MAX_DOC_LEN,                                      padding='post',                                      truncating='post')
    X_train, X_test, Y_train, Y_test = train_test_split(                            padded_sequences, text['label'],                            test_size=0.2, random_state=1)
    main_input = Input(shape=(MAX_DOC_LEN,),                        dtype='int32', name='main_input')
    embed_1 = Embedding(input_dim=MAX_NB_WORDS+1,                     output_dim=EMBEDDING_DIM,                     input_length=MAX_DOC_LEN,                    name='embedding')(main_input)
    conv1d_1= Conv1D(filters=32, kernel_size=1,                      name='conv_unigram',                     activation='relu')(embed_1)
    pool_1 = MaxPooling1D(MAX_DOC_LEN-1+1,name='pool_unigram')(conv1d_1)  # unigram -1 
    flat_1 = Flatten(name='flat_unigram')(pool_1)
    conv1d_2= Conv1D(filters=64, kernel_size=2,                      name='conv_bigram',                     activation='relu')(embed_1)
    pool_2 = MaxPooling1D(MAX_DOC_LEN-2+1, name='pool_bigram')(conv1d_2)
    flat_2 = Flatten(name='flat_bigram')(pool_2)
    conv1d_3= Conv1D(filters=64, kernel_size=3,                      name='conv_trigram',activation='relu')(embed_1)
    pool_3 = MaxPooling1D(MAX_DOC_LEN-3+1, name='pool_trigram')(conv1d_3)
    flat_3 = Flatten(name='flat_trigram')(pool_3)
    z=Concatenate(name='concate')([flat_1, flat_3])
    drop_1=Dropout(rate=0.5, name='dropout')(z)
    dense_1 = Dense(192, activation='relu', name='dense')(drop_1)
    preds = Dense(1, activation='sigmoid', name='output')(dense_1)
    model = Model(inputs=main_input, outputs=preds)
    model.compile(loss="binary_crossentropy",               optimizer="adam",               metrics=["accuracy"])
    training=model.fit(X_train, Y_train-1,                        batch_size=BATCH_SIZE,                        epochs=NUM_EPOCHES,                       validation_data=[X_test, Y_test-1],                        verbose=2)
    df=pd.DataFrame.from_dict(training.history)
    df.columns=["train_acc", "train_loss",                 "val_acc", "val_loss"]
    df.index.name='epoch'
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,3));
    df[["train_acc", "val_acc"]].plot(ax=axes[0])
    df[["train_loss", "val_loss"]].plot(ax=axes[1])
    plt.show()
    prediction=model.predict(X_test)
    prediction=np.where(prediction>0.5, 2, 1)
    print(classification_report(Y_test, prediction))

    
if __name__ == "__main__":
    text = pd.read_csv("amazon_review_500.csv")
    sentiment_cnn(text)

