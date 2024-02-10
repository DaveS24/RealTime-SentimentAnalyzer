# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


## [0.2.1] - 2024-02-10

### Added

- The data_preparation.py file to handle all of the data preparation steps.
- The data_analysis.ipynb file to analyze the dataset.
- The rnn_model.py file to create the RNN model.
- Comparing results of the RNN model across different input-data in the rnn_analysis.ipynb file.

### Changed

- Renamed the rnn_model.ipynb file to rnn_analysis.ipynb and moved the RNN model creation and training to the rnn_model.py file.
- Data loading and preprocessing steps to be handled in the data_preparation.py file.
- Data analysis steps to be handled in the data_analysis.ipynb file.

### Removed

- The data_loader.ipynb file.
- The data_preprocessor.ipynb file.


## [0.2.0] - 2024-02-09

### Added

- The rnn_model.ipynb file to create the RNN model.
- Building the basic RNN model in the rnn_model.ipynb file.
- Training the RNN model in the rnn_model.ipynb file.
- Evaluating the RNN model in the rnn_model.ipynb file.

### Fixed

- The train-test-dev split in the data_preprocessor.ipynb file.


## [0.1.2] - 2024-02-09

### Added

- Removal of stopwords from the dataset in the data_preprocessor.ipynb file.
- Analysis of the dataset in the data_loader.ipynb file by
    - visualizing basic statistics.
    - visualizing the sentiment distribution.
    - visualizing the word cloud.
    - visualizing the embeddings using t-SNE.
- Tokenization of the dataset in the data_preprocessor.ipynb file.
- Splitting the dataset into training and testing sets in the data_preprocessor.ipynb file.


## [0.1.1] - 2024-02-08

### Added

- The data_loader.ipynb file to load the dataset.
- The data_preprocessor.ipynb file to preprocess the dataset.

- Loading the dataset in the data_loader.ipynb file.
- Dropping missing values from the dataset in the data_preprocessor.ipynb file.
- Preprocessing the dataset in the data_preprocessor.ipynb file using the `nltk` library to
    - tokenize.
    - remove punctuation.
    - perform stemming or lemmatization.


## [0.1.0] - 2024-02-08

### Added

- This CHANGELOG file.
- The .gitignore file.
- The stanfordSentimentTreebank dataset.
