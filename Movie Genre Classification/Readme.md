# Movie Genre Classification

## Overview

Welcome to the Movie Genre Classification model repository! This project aims to classify movies into different genres using machine learning techniques. The model is trained on a dataset of movie features and labels to predict the genre of a given movie.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **To get started, clone this repository to your local machine:**

  ```bash
  git clone https://github.com/Dharaneesh-23/Movie-Genre-Classification.git
  ```

2. **Navigate to the folder:**
    ```bash
   cd Movie-Genre-Classification
   

3. **Install required packages:**
    ```bash
    pip install sckit-learn
    pip install pandas
    pip install numpy
## Usage
1. **Pre-trained Model**
    If you just want to use the pre-trained model using SVM for inference, you can do so by running:
    ```bash
    python GenreClassificationSVM.ipynb

2. **Train Your Model**
  If you want to train your own model, follow the instructions in [Model Training](#model-training) section.

## Dataset
The model is trained on a dataset of movies with the following features:

| Title                             | Year | Genre      | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|-----------------------------------|------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Oscar et la dame rose (2009)       | 2009 | Drama      | Listening in to a conversation between his doctor and parents, 10-year-old Oscar learns what nobody has the courage to tell him. He only has a few weeks to live. Furious, he refuses to speak to anyone except straight-talking Rose, the lady in pink he meets on the hospital stairs. As Christmas approaches, Rose uses her fantastical experiences as a professional wrestler, her imagination, wit and charm to allow Oscar to live life and love to the full, in the company of his friends Pop Corn, Einstein, Bacon and childhood sweetheart Peggy Blue. |
| Cupid (1997)                       | 1997 | Thriller   | A brother and sister with a past incestuous relationship have a current murderous relationship. He murders the women who reject him and she murders the women who get too close to him.                                                                                                                                                                                                                                                                                                                      |

This is the sample form of the dataset. To download the dataset [Click here](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)

## Model Training
To train the model, use the following command:
  1. Data Collection and Preparation:
    - Collect Data: Gather a dataset of movie information, including features (e.g., movie descriptions) and labels (genres).
    - Data Cleaning: Clean the data by handling missing values, removing duplicates, and addressing any other data quality issues.
    - Data Preprocessing: Convert text data into a format suitable for machine learning, such as tokenization and vectorization.

  2. Exploratory Data Analysis (EDA):
    - Explore the Data: Analyze the dataset to understand its characteristics, distribution, and relationships.
    - Visualize Data: Use plots and charts to visualize trends, patterns, and potential correlations in the data.

  3. Feature Engineering:
    - Text Vectorization: Convert movie descriptions into numerical vectors using techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings.
    - Additional Features: Consider adding other relevant features that might improve model performance.

  4. Model Selection:
    - Choose Model Architecture: Select a suitable model architecture for text classification. Common choices include recurrent neural networks (RNNs), long short-term memory networks (LSTMs), or transformer-based models like BERT.
    - Compile Model: Specify the loss function, optimizer, and evaluation metric for your model.

  5. Model Training:
    - Split Data: Divide the dataset into training and validation sets.
    - Train Model: Fit the model to the training data using the training set, validating its performance on the validation set.
    - Hyperparameter Tuning: Adjust hyperparameters based on model performance.

  6. Evaluation:
    - Evaluate Model: Assess the model's performance on a separate test dataset.
    - Metrics: Use metrics like accuracy, precision, recall, and F1 score for classification models.
  
  7. Model Deployment (Optional):
    - Deploy Model: If you want to use the model for predictions, deploy it to a production environment.
     
## Evaluation
Evaluate the model on a test dataset using:
  - Extract the dataset from the test dataset and preprocess for the required features.
  - Pass the features to model trained to get the predicitons.
  - Compare the predictions with the actual values to get the accuracy of the model.

## Results
  Obatained an accuracy of the 63.5% with the SVM classifier model. 
  As the model is in basic stage it can be developed to implement in the webpages or any other cases.
