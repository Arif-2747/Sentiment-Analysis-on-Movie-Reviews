import pandas as pd
import numpy as np
import nltk 
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Ensure required NLTK datasets are downloaded
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Define stop_words globally
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocess function (with NaN/type handling)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Load the movie reviews dataset
def load_movie_reviews():
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            document = movie_reviews.raw(fileid)
            documents.append((document, category))
    return documents

# Load data
documents = load_movie_reviews()

# Convert to DataFrame
df = pd.DataFrame(documents, columns=['review', 'sentiment'])

# Check for NaN values and handle them
df = df.dropna(subset=['review'])  # Remove rows with NaN in 'review'
df['review'] = df['review'].astype(str)  # Convert all entries to string

# Preprocess text
df['review'] = df['review'].apply(preprocess_text)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# --- Generate the summary graph ---

# Extract metrics for 'neg' and 'pos' classes
classes = ['neg', 'pos']
metrics = ['precision', 'recall', 'f1-score']

# Prepare data for plotting
report = classification_report(y_test, y_pred, output_dict=True)
neg_scores = [report['neg'][metric] for metric in metrics]
pos_scores = [report['pos'][metric] for metric in metrics]

x = np.arange(len(metrics)) # the label locations
width = 0.35 # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6)) # Adjust figure size for better readability

# Create bars for 'neg' and 'pos' sentiments
rects1 = ax.bar(x - width/2, neg_scores, width, label='Negative', color='red')
rects2 = ax.bar(x + width/2, pos_scores, width, label='Positive', color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Score')
ax.set_title('Classification Report Metrics by Sentiment')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.1) # Set y-axis limit from 0 to 1.1 for better visualization

# Function to add labels on top of the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show() # Display the plot
