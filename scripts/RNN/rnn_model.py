import data_preparation as dp

import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.metrics import CategoricalAccuracy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


MAX_FEATURES = 10000
MAX_SEQUENCE_LENGTH = 100


def build_model():
    model = Sequential()
    model.add(Embedding(MAX_FEATURES, 128, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(128, dropout=0.5, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(64, dropout=0.5, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[CategoricalAccuracy()])
    
    return model

def train_and_evaluate(sentiment_data_df, column_method='tokenized_text'):
    # Receive the train, test, and dev data
    train_data, test_data, dev_data = dp.train_test_dev_split(sentiment_data_df, column_method)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    # Build and train the model
    model = build_model()
    model.fit(train_data[0], train_data[1], epochs=20, batch_size=32, validation_data=(dev_data[0], dev_data[1]), callbacks=[early_stopping])

    # Evaluate the model
    test_predictions = model.predict(test_data[0])
    test_predictions_classes = np.argmax(test_predictions, axis=1)
    test_true_classes = np.argmax(test_data[1], axis=1)

    accuracy = accuracy_score(test_true_classes, test_predictions_classes)
    precision = precision_score(test_true_classes, test_predictions_classes, average='weighted')
    recall = recall_score(test_true_classes, test_predictions_classes, average='weighted')
    f1 = f1_score(test_true_classes, test_predictions_classes, average='weighted')

    return accuracy, precision, recall, f1