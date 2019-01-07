import tensorflow as tf
import pandas as pd
from pandas import DataFrame
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM

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

    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

    # Build Model
    print('Build model...')
    model = Sequential()
    model.add(LSTM(64, input_shape=(154, 3), dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(26, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    model.summary()

    # Train Model
    print('Train...')
    model.fit(train_x, train_y,
            batch_size=30,
            epochs=15,
            validation_data=(test_x, test_y))
    score, acc = model.evaluate(test_x, test_y,
                                batch_size=30)
    
    # Test Model
    print('Test score:', score)
    print('Test accuracy:', acc)

    '''
    ...
    Epoch 15/15
    1375/1375 [==============================] - 6s 4ms/step - loss: 3.2560 - acc: 0.0436 - val_loss: 3.2639 - val_acc: 0.0383
    678/678 [==============================] - 1s 935us/step
    Test score: 3.263940857575003
    Test accuracy: 0.03834808413433818
    '''