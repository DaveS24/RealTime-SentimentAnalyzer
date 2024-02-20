# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


## [Unreleased]


## [0.4.1] - 2024-02-20

### Added

- Class weights to the RNN model to handle the class imbalance in the rnn_model.py file.
- The rnn_results.csv file to store the results for different RNN model-hyperparameters.
- Confusion matrices for each RNN model in the rnn_analysis.ipynb file.

### Changed

- Updated the README.md file to include the new classification task.

### Fixed

- A plotting issue in the data_analysis.ipynb file.


## [0.4.0] - 2024-02-19

### Changed

- Transitioned the problem from a regression task to a five-class classification task.
    - The five classes are: very negative, negative, neutral, positive, and very positive.
    - Updated the data_preparation.py file to handle the new classification task.
    - Updated the data_analysis.ipynb file to analyze the new classification task.


## [0.3.0] - 2024-02-19

### Added

- The transformer_analysis.ipynb file to analyze the transformer model.
- The transformer_model.py file to create the transformer model.

### Changed

- Structure of the project to be more modular.

### Fixed

- The .gitignore file to ignore the .vscode directory.
- Some importing issues in both the rnn_model.py and transformer_model.py files.
- An issue with downloading the necessary nltk packages in the data_preparation.py file.


## [0.2.2] - 2024-02-16

### Added

- Set a fixed seed for reproducibility in the rnn_model.py file.

### Changed

- The hyperparameters of the RNN model to reduce overfitting.
- Testing out multiple layers to the RNN model in the rnn_model.py file.

### Fixed

- A .gitignore issue where the __pycache__ directory was not being ignored.


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
