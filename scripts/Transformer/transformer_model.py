import scripts.data_preparation as dp

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
    model, tokenizer = build_model()
    train_encodings = tokenizer(train_data[0].tolist(), truncation=True, padding=True, max_length=MAX_SEQUENCE_LENGTH)
    dev_encodings = tokenizer(dev_data[0].tolist(), truncation=True, padding=True, max_length=MAX_SEQUENCE_LENGTH)
    model.fit([train_encodings['input_ids'], train_encodings['attention_mask']], train_data[1], epochs=20, batch_size=32, validation_data=([dev_encodings['input_ids'], dev_encodings['attention_mask']], dev_data[1]), callbacks=[early_stopping])

    # Evaluate the model
    test_encodings = tokenizer(test_data[0].tolist(), truncation=True, padding=True, max_length=MAX_SEQUENCE_LENGTH)
    test_predictions = model.predict([test_encodings['input_ids'], test_encodings['attention_mask']])
    rmse = sqrt(mean_squared_error(test_data[1], test_predictions))
    r2 = r2_score(test_data[1], test_predictions)

    return rmse, r2