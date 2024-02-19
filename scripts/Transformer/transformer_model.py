import data_preparation as dp

import numpy as np

from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score

from transformers import TFDistilBertModel, DistilBertConfig, DistilBertTokenizerFast
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling1D, Dense, Dropout
from keras.callbacks import EarlyStopping


MAX_FEATURES = 10000
MAX_SEQUENCE_LENGTH = 100


def build_model():
    # Define the DistilBERT tokenizer and model
    distil_bert = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizerFast.from_pretrained(distil_bert)
    transformer_model = TFDistilBertModel.from_pretrained(distil_bert)

    # Define the input layers
    input_ids = Input(shape=(MAX_SEQUENCE_LENGTH,), name='input_token', dtype='int32')
    input_masks = Input(shape=(MAX_SEQUENCE_LENGTH,), name='masked_token', dtype='int32')

    # Define the transformer model and connect it with the inputs
    embedding = transformer_model(input_ids, attention_mask=input_masks)[0]
    X = GlobalAveragePooling1D()(embedding)
    X = Dense(64, activation='relu')(X)
    X = Dropout(0.2)(X)
    X = Dense(1)(X)

    # Define the model
    model = Model(inputs=[input_ids, input_masks], outputs = X)

    for layer in model.layers[:3]:
        layer.trainable = False

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model, tokenizer


def train_and_evaluate(sentiment_data_df, column_method='tokenized_text'):
    # Receive the train, test, and dev data
    train_data, test_data, dev_data = dp.train_test_dev_split(sentiment_data_df, column_method)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    
    # Build and train the model
    model, _ = build_model()  # tokenizer is not needed
    train_encodings = train_data[0]
    dev_encodings = dev_data[0]
    model.fit([train_encodings, np.ones(train_encodings.shape)], train_data[1], epochs=20, batch_size=32, validation_data=([dev_encodings, np.ones(dev_encodings.shape)], dev_data[1]), callbacks=[early_stopping])

    # Evaluate the model
    test_encodings = test_data[0]
    test_predictions = model.predict([test_encodings, np.ones(test_encodings.shape)])
    rmse = sqrt(mean_squared_error(test_data[1], test_predictions))
    r2 = r2_score(test_data[1], test_predictions)

    return rmse, r2