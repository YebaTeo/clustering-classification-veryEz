import pandas as pd
import re
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv('twitter_training.csv', header=None, names=['Tweet_ID', 'Entity', 'Sentiment', 'Tweet'])

# Optional: Filter out 'Irrelevant' sentiments
df = df[df['Sentiment'] != 'Irrelevant']
df['Tweet'] = df['Tweet'].fillna('')

# Text preprocessing
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['Clean_Tweet'] = df['Tweet'].apply(clean_text)

# Feature extraction with TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Clean_Tweet'])
y = df['Sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=== Logistic Regression Classification Report ===")
print(classification_report(y_test, y_pred))

# WordCloud Function
def plot_wordcloud(sentiment_label):
    text = ' '.join(df[df['Sentiment'] == sentiment_label]['Clean_Tweet'].dropna())
    wordcloud = WordCloud(width=500, height=300, background_color='white', colormap='Set2').generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment_label} Tweets')
    plt.show()

for sentiment in df['Sentiment'].unique():
    plot_wordcloud(sentiment)

# === k-NN Classifier with Euclidean Distance ===
knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X_train, y_train)
y_knn_pred = knn_model.predict(X_test)

print("=== k-NN Classification Report (Euclidean Distance) ===")
print(classification_report(y_test, y_knn_pred))

# Plot function for k-NN prediction in 2D using PCA
def plot_knn_predictions(X, y_true, y_pred, title='k-NN Predictions (PCA)'):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X.toarray())

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=pd.factorize(y_pred)[0], cmap='Set2', alpha=0.7)
    plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)

    # Correct legend handling
    handles, _ = scatter.legend_elements()
    unique_labels = list(pd.Series(y_pred).unique())
    plt.legend(handles=handles, labels=unique_labels, title="Sentiment")
    plt.show()

# Plot the results
plot_knn_predictions(X_test, y_test, y_knn_pred)
