import data_preparation as dp

from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import EarlyStopping


MAX_FEATURES = 10000
MAX_SEQUENCE_LENGTH = 100


def build_model():
    model = Sequential()
    model.add(Embedding(MAX_FEATURES, 256, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(256, dropout=0.5, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(128, dropout=0.5, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(64, dropout=0.5, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error',
                  optimizer='adam')
    
    return model


def train_and_evaluate(sentiment_data_df, column_method='tokenized_text'):
    # Receive the train, test, and dev data
    train_data, test_data, dev_data = dp.train_test_dev_split(sentiment_data_df, column_method)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    # Build and train the model
    model = build_model()
    model.fit(train_data[0], train_data[1], epochs=20, batch_size=32, validation_data=(dev_data[0], dev_data[1]), callbacks=[early_stopping])

    # Evaluate the model
    test_predictions = model.predict(test_data[0])
    rmse = sqrt(mean_squared_error(test_data[1], test_predictions))
    r2 = r2_score(test_data[1], test_predictions)

    return rmse, r2