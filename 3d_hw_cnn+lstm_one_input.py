import tensorflow as tf
import pandas as pd
from pandas import DataFrame
import numpy as np

from tensorflow.keras.models import Sequential, Model, save_model
from tensorflow.keras.layers import Input, Dense, Embedding, BatchNormalization
from tensorflow.keras.layers import Conv1D, LSTM, MaxPooling1D, Dropout, Concatenate, Activation, concatenate

from sklearn import model_selection

def get_dataframe_from_excel(fpath, sheetname=0):
    df = pd.read_excel(fpath, sheetname, header=None)
    return df

def make_train_test_split(train, answer):
    test_size = 0.33
    seed = 7
    return model_selection.train_test_split(train, answer, test_size=test_size, random_state=seed)

if __name__ == "__main__":

    # Make Dataset
    hand_writing = {
        'dataset': './3D_handwriting_train.xlsx',
        'train': 'train_data',
        'answer': 'answer'
    }

    train_X = get_dataframe_from_excel(hand_writing['dataset'], 'Acc-X')
    train_Y = get_dataframe_from_excel(hand_writing['dataset'], 'Acc-Y')
    train_Z = get_dataframe_from_excel(hand_writing['dataset'], 'Acc-Z')
    hand_writing_answer = get_dataframe_from_excel(hand_writing['dataset'], hand_writing['answer'])

    print(train_X.shape, train_Y.shape, train_Z.shape, hand_writing_answer.shape)

    # fearture padding, answer encoding
    train_X = train_X.fillna(0).values
    train_Y = train_Y.fillna(0).values
    train_Z = train_Z.fillna(0).values

    hand_writing_answer = hand_writing_answer.applymap(lambda x: ord(x) - 97).values
    hand_writing_answer = tf.keras.utils.to_categorical(hand_writing_answer, 26)

    train = np.dstack([train_X, train_Y, train_Z])

    train_x, test_x, train_y, test_y = make_train_test_split(train[:1900], hand_writing_answer[:1900])

    # Build Model
    # batch_size = 30
    batch_size = 5
    # epochs = 100
    epochs = 10
    kernel_size = 3
    pool_size = 2
    dropout_rate = 0.15
    f_act = 'relu'
    n_classes = 26

    inputs = Input(shape=(train_x.shape[1], train_x.shape[2]))
    acc = Conv1D(512, (kernel_size), activation=f_act, padding='same')(inputs)
    acc = BatchNormalization()(acc)
    acc = MaxPooling1D(pool_size=(pool_size))(acc)
    acc = Dropout(dropout_rate)(acc)
    acc = Conv1D(64, (kernel_size), activation=f_act, padding='same')(acc)
    acc = BatchNormalization()(acc)
    acc = MaxPooling1D(pool_size=(pool_size))(acc)
    acc = Dropout(dropout_rate)(acc)
    acc = Conv1D(64, (kernel_size), activation=f_act, padding='same')(acc)
    acc = BatchNormalization()(acc)
    acc = MaxPooling1D(pool_size=(pool_size))(acc)
    acc = LSTM(128, return_sequences=True)(acc)
    acc = LSTM(128, return_sequences=True)(acc)
    acc = LSTM(128)(acc)
    acc = Dropout(dropout_rate)(acc)
    acc = Dense(n_classes)(acc)
    acc = BatchNormalization()(acc)
    output = Activation('softmax')(acc)

    main_model = Model(inputs=inputs, outputs=[output])

    main_model.summary()

    # Train Model
    main_model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

    history = main_model.fit(train_x, train_y,
                    batch_size=batch_size, epochs=epochs,
                    validation_data=(test_x, test_y))

    # score, acc = main_model.evaluate(test_x, test_y,
    #                             batch_size=30)
    # score, acc = main_model.evaluate(train[1900:], hand_writing_answer[1900:], batch_size=30)
    score, acc = main_model.evaluate(train[1900:], hand_writing_answer[1900:], batch_size=5)
    print('Test score:', score)
    print('Test accuracy:', acc)

    '''
    Epoch 100/100
    1375/1375 [==============================] - 6s 4ms/step - loss: 0.0810 - acc: 0.9876 - val_loss: 0.8284 - val_acc: 0.8687
    678/678 [==============================] - 1s 814us/step
    Test score: 0.8283678647954907
    Test accuracy: 0.8687315519932097
    '''