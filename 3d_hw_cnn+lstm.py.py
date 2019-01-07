import tensorflow as tf
import pandas as pd
from pandas import DataFrame
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Embedding, BatchNormalization
from tensorflow.keras.layers import Conv1D, LSTM, MaxPooling1D, Dropout, Activation, concatenate

from sklearn import model_selection

def get_dataframe_from_excel(fpath, sheetname=0):
    df = pd.read_excel(fpath, sheetname)
    return df

def make_train_test_split(train, answer):
    test_size = 0.33
    seed = 7
    return model_selection.train_test_split(train, answer, test_size=test_size, random_state=seed)

if __name__ == "__main__":

    # Make Dataset
    hand_writing = {
        'dataset': './data/3D_handwriting_train_new.xlsx',
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

    train_x, test_x, train_y, test_y = make_train_test_split(train, hand_writing_answer)

    train_x = np.dsplit(train_x, 3)
    test_x = np.dsplit(test_x, 3)

    # Build Model
    batch_size = 30
    epochs = 100
    kernel_size = 3
    pool_size = 2
    dropout_rate = 0.15
    f_act = 'relu'
    n_classes = 26

    inputs_x = Input(shape=(train_x[0].shape[1], train_x[0].shape[2]))
    acc_x = Conv1D(512, (kernel_size), activation=f_act, padding='same')(inputs_x)
    acc_x = BatchNormalization()(acc_x)
    acc_x = MaxPooling1D(pool_size=(pool_size))(acc_x)
    acc_x = Dropout(dropout_rate)(acc_x)
    acc_x = Conv1D(64, (kernel_size), activation=f_act, padding='same')(acc_x)
    acc_x = BatchNormalization()(acc_x)
    acc_x = MaxPooling1D(pool_size=(pool_size))(acc_x)
    acc_x = Dropout(dropout_rate)(acc_x)
    acc_x = Conv1D(64, (kernel_size), activation=f_act, padding='same')(acc_x)
    acc_x = BatchNormalization()(acc_x)
    acc_x = MaxPooling1D(pool_size=(pool_size))(acc_x)
    acc_x = LSTM(128, return_sequences=True)(acc_x)
    acc_x = LSTM(128, return_sequences=True)(acc_x)
    acc_x = LSTM(128)(acc_x)
    output_x = Dropout(dropout_rate)(acc_x)
    first_model = Model(inputs=inputs_x, outputs=output_x)


    inputs_y = Input(shape=(train_x[0].shape[1], train_x[0].shape[2]))
    acc_y = Conv1D(512, (kernel_size), activation=f_act, padding='same')(inputs_y)
    acc_y = BatchNormalization()(acc_y)
    acc_y = MaxPooling1D(pool_size=(pool_size))(acc_y)
    acc_y = Dropout(dropout_rate)(acc_y)
    acc_y = Conv1D(64, (kernel_size), activation=f_act, padding='same')(acc_y)
    acc_y = BatchNormalization()(acc_y)
    acc_y = MaxPooling1D(pool_size=(pool_size))(acc_y)
    acc_y = Dropout(dropout_rate)(acc_y)
    acc_y = Conv1D(64, (kernel_size), activation=f_act, padding='same')(acc_y)
    acc_y = BatchNormalization()(acc_y)
    acc_y = MaxPooling1D(pool_size=(pool_size))(acc_y)
    acc_y = LSTM(128, return_sequences=True)(acc_y)
    acc_y = LSTM(128, return_sequences=True)(acc_y)
    acc_y = LSTM(128)(acc_y)
    output_y = Dropout(dropout_rate)(acc_y)
    second_model = Model(inputs=inputs_y, outputs=output_y)

    inputs_z = Input(shape=(train_x[0].shape[1], train_x[0].shape[2]))
    acc_z = Conv1D(512, (kernel_size), activation=f_act, padding='same')(inputs_z)
    acc_z = BatchNormalization()(acc_z)
    acc_z = MaxPooling1D(pool_size=(pool_size))(acc_z)
    acc_z = Dropout(dropout_rate)(acc_z)
    acc_z = Conv1D(64, (kernel_size), activation=f_act, padding='same')(acc_z)
    acc_z = BatchNormalization()(acc_z)
    acc_z = MaxPooling1D(pool_size=(pool_size))(acc_z)
    acc_z = Dropout(dropout_rate)(acc_z)
    acc_z = Conv1D(64, (kernel_size), activation=f_act, padding='same')(acc_z)
    acc_z = BatchNormalization()(acc_z)
    acc_z = MaxPooling1D(pool_size=(pool_size))(acc_z)
    acc_z = LSTM(128, return_sequences=True)(acc_z)
    acc_z = LSTM(128, return_sequences=True)(acc_z)
    acc_z = LSTM(128)(acc_z)
    output_z = Dropout(dropout_rate)(acc_z)
    third_model = Model(inputs=inputs_z, outputs=output_z)

    acc = concatenate([output_x, output_y, output_z])
    acc = Dropout(0.4)(acc)
    acc = Dense(n_classes)(acc)
    acc = BatchNormalization()(acc)
    output = Activation('softmax')(acc)
    main_model = Model(inputs=[inputs_x, inputs_y, inputs_z], outputs=[output])

    main_model.summary()
    
    # Train Model
    main_model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

    history = main_model.fit([train_x[0], train_x[1], train_x[2]], train_y,
                    batch_size=batch_size, epochs=epochs,
                    validation_data=([test_x[0], test_x[1], test_x[2]], test_y))

    score, acc = main_model.evaluate([test_x[0], test_x[1], test_x[2]], test_y,
                                batch_size=30)

    # Test Model
    print('Test score:', score)
    print('Test accuracy:', acc)

    '''
    ...
    Epoch 100/100
    1375/1375 [==============================] - 16s 11ms/step - loss: 0.0986 - acc: 0.9869 - val_loss: 1.2917 - val_acc: 0.6844
    678/678 [==============================] - 1s 2ms/step
    Test score: 1.2917028027298176
    Test accuracy: 0.6843657846999379
    '''