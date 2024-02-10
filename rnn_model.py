import data_preparation as dp

from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense


MAX_FEATURES = 10000
MAX_SEQUENCE_LENGTH = 100


def build_model():
    model = Sequential()
    model.add(Embedding(MAX_FEATURES, 128, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error',
              optimizer='adam')
    
    return model


def train_and_evaluate(sentiment_data_df, column_method='tokenized_text'):
    # Receive the train, test, and dev data
    train_data, test_data, dev_data = dp.train_test_dev_split(sentiment_data_df, column_method)

    # Build and train the model
    model = build_model()
    model.fit(train_data[0], train_data[1], epochs=10, batch_size=32, validation_data=(dev_data[0], dev_data[1]))

    # Evaluate the model
    test_predictions = model.predict(test_data[0])
    rmse = sqrt(mean_squared_error(test_data[1], test_predictions))
    r2 = r2_score(test_data[1], test_predictions)

    return rmse, r2