import pandas as pd
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


MAX_FEATURES = 10000
MAX_SEQUENCE_LENGTH = 100


def data_loader(data_prefix):
    # Load all dataframes
    sentences_df = pd.read_csv(data_prefix + 'datasetSentences.txt', sep='\t')
    dictionary_df = pd.read_csv(data_prefix + 'dictionary.txt', sep='|', names=['phrase', 'phrase ids'])
    sentiment_labels_df = pd.read_csv(data_prefix + 'sentiment_labels.txt', sep='|')
    dataset_split_df = pd.read_csv(data_prefix + 'datasetSplit.txt', sep=',')

    # Merge all dataframes
    sentiment_data_df = (sentences_df
                         .merge(dictionary_df, left_on='sentence', right_on='phrase', how='left')
                         .merge(sentiment_labels_df, on='phrase ids', how='left')
                         .merge(dataset_split_df, on='sentence_index', how='left'))
    
    # Drop missing values
    sentiment_data_df = sentiment_data_df.dropna()

    # Convert sentiment scores to class labels
    sentiment_data_df['sentiment_values'] = pd.cut(sentiment_data_df['sentiment_values'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=False, include_lowest=True)
    
    return sentiment_data_df


def data_preprocessor(sentiment_data_df):
    # Download necessary nltk data if not already available
    if not nltk.download('punkt', download_dir=nltk.data.path[0], quiet=True):
        nltk.download('punkt')
    if not nltk.download('stopwords', download_dir=nltk.data.path[0], quiet=True):
        nltk.download('stopwords')
    if not nltk.download('wordnet', download_dir=nltk.data.path[0], quiet=True):
        nltk.download('wordnet')

    # Tokenize the text data
    sentiment_data_df['tokenized_text'] = sentiment_data_df['sentence'].str.lower().apply(word_tokenize)

    # Remove punctuation
    sentiment_data_df['no_punctuation_text'] = sentiment_data_df['tokenized_text'].apply(lambda x: [word for word in x if word not in string.punctuation])

    # Remove stopwords
    custom_stopwords = set(stopwords.words('english')) - {'but', 'not', 'no', 'nor'}
    sentiment_data_df['no_stopwords_text'] = sentiment_data_df['tokenized_text'].apply(lambda x: [word for word in x if word not in custom_stopwords])

    # Perform stemming
    stemmer = PorterStemmer()
    sentiment_data_df['stemmed_text'] = sentiment_data_df['tokenized_text'].apply(lambda x: [stemmer.stem(word) for word in x])

    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    sentiment_data_df['lemmatized_text'] = sentiment_data_df['tokenized_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    return sentiment_data_df


def train_test_dev_split(sentiment_data_df, split_column='tokenized_text'):
    # Tokenize the text
    tokenizer = Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(sentiment_data_df[split_column].apply(lambda x: ' '.join(x)))
    sequences = tokenizer.texts_to_sequences(sentiment_data_df[split_column].apply(lambda x: ' '.join(x)))

    sentiment_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    train_mask = sentiment_data_df['splitset_label'] == 1
    test_mask = sentiment_data_df['splitset_label'] == 2
    dev_mask = sentiment_data_df['splitset_label'] == 3

    # Convert integer labels to one-hot encoded labels
    sentiment_values_one_hot = to_categorical(sentiment_data_df['sentiment_values'])

    train_data = (sentiment_data[train_mask], sentiment_values_one_hot[train_mask])
    test_data = (sentiment_data[test_mask], sentiment_values_one_hot[test_mask])
    dev_data = (sentiment_data[dev_mask], sentiment_values_one_hot[dev_mask])

    return train_data, test_data, dev_data
