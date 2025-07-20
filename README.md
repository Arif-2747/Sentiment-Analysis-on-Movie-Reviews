Sentiment Analysis on Movie Reviews
===================================
This repository contains a Python script for performing sentiment analysis on movie reviews using a Logistic Regression model. The script preprocesses text data, converts it into numerical features using TF-IDF, trains a classification model, and evaluates its performance, including a visual summary of key metrics.

Description
-----------------
This project implements a basic sentiment analysis pipeline. It leverages the NLTK movie_reviews dataset, which contains movie reviews categorised as either 'positive' or 'negative'. The text data undergoes preprocessing steps such as tokenisation, lowercasing, stop word removal, and stemming. A TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is used to transform the text into numerical features, which are then fed into a Logistic Regression model for classification. Finally, the model's performance is evaluated, and a bar chart visualising precision, recall, and F1-score for both sentiment classes is generated.

Examples:

 Original Movie Review (from the movie_reviews dataset):
"The movie was absolutely fantastic! The acting was superb and the story was truly engaging."

**1. Preprocessing:**

Tokenization & Lowercasing: ["the", "movie", "was", "absolutely", "fantastic", "!", "the", "acting", "was", "superb", "and", "the", "story", "was", "truly", "engaging", "."]

Stop Word Removal & Punctuation Removal: ["movie", "absolutely", "fantastic", "acting", "superb", "story", "truly", "engaging"]

Stemming (Porter Stemmer): ["movi", "absolut", "fantast", "act", "superb", "stori", "truli", "engag"]

Final Preprocessed Text: "movi absolut fantast act superb stori truli engag"

**2. TF-IDF Vectorisation:**
This preprocessed text is then converted into a numerical vector by the TfidfVectorizer. Each dimension in this vector corresponds to a unique word in the entire dataset's vocabulary, and the value in that dimension reflects the TF-IDF score of that word for this specific review. For example, words like "fantast" and "superb" would likely have higher TF-IDF scores for a positive review.

**3. Logistic Regression Classification:**
The numerical TF-IDF vector representing the review is fed into the trained Logistic Regression model. The model calculates the probability of the review belonging to the 'positive' class and the 'negative' class. Based on these probabilities (e.g., if the probability for 'positive' is > 0.5), it assigns a predicted sentiment.

**4. Predicted Sentiment:**
For this example review, the model would likely predict: Positive

This process is repeated for all reviews in the dataset, and the aggregated predictions are then used for model evaluation and generating the classification report and the bar chart.

Features 
-----------------

**Text Preprocessing:** Includes tokenisation, lowercasing, stop word removal, and Porter stemming.

**TF-IDF Vectorisation:** Converts text data into numerical feature vectors.

**Logistic Regression Model:** A simple yet effective classification model for sentiment prediction.

**Model Evaluation:** Calculates and prints accuracy and a detailed classification report.

**Visual Summary:** Generates a bar chart showing precision, recall, and F1-score for positive and negative sentiments.

**Handles Missing Data:** Basic handling for non-string or NaN values in the input text.

Prerequisites
--------------
Before running the script, ensure you have the following installed:

  - Python 3.x

  - pip (Python package installer)

Installation
-------------
**1. Install required Python libraries:**
      
      pip install pandas numpy scikit-learn nltk matplotlib


**2. Download NLTK datasets:**

The script automatically attempts to download the necessary NLTK datasets (movie_reviews, stopwords, punkt, punkt_tab). However, if you encounter any issues, you can manually download them by running a Python interpreter and executing:

      import nltk
      nltk.download('movie_reviews')
      nltk.download('stopwords')
      nltk.download('punkt')


Usage
-----------
To run the sentiment analysis script, simply execute the Python file from your terminal:

      python sentiment_analysis.py


Output
---------
Upon execution, the script will:

  1. Print the overall accuracy of the Logistic Regression model on the test set.

  2. Print a detailed classification report, including precision, recall, F1-score, and support for both 'negative' and 'positive' classes.

  3. Display a bar chart titled "Classification Report Metrics by Sentiment", visually comparing the precision, recall, and F1-score for negative (red bars) and positive (green bars) sentiments.

Example of console output:

    Accuracy: 0.815
    Classification Report:
              precision    recall  f1-score   support

         neg       0.81      0.82      0.82       196
         pos       0.82      0.81      0.81       204

    accuracy                           0.82       400
    macro avg       0.82      0.82      0.82       400
    weighted avg       0.82      0.82      0.82       400


(A matplotlib graph window will also pop up.)

Model Details
---------------
**TF-IDF Vectorisation**

TF-IDF stands for Term Frequency-Inverse Document Frequency. It's a numerical statistic that reflects the importance of a word in a document within a collection or corpus. It is often used as a weighting factor in information retrieval and text mining. The TF-IDF value increases proportionally to the number of times a word appears in the document. It is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general.

**Logistic Regression**

Logistic Regression is a statistical model that, in its basic form, uses a logistic function to model a binary dependent variable. However, it can be extended to model more than two outcomes (multinomial logistic regression). In this project, it is used as a classification algorithm to predict whether a movie review is positive or negative. Despite its name, Logistic Regression is a classification algorithm, not a regression algorithm.

Contributing
---------------
You can fix this repository by opening issues and submitting pull requests.

License
---------------
This project is open-source and available under the MIT License.
