import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Sample Data (If you have your own dataset, replace this part)
# Sample dataset of tweets related to a specific topic or brand
data = {
    'tweet': [
        "I love the new design of the iPhone!",
        "The battery life of this phone is terrible.",
        "Amazing camera quality. Highly recommend!",
        "I hate the user interface, it's so confusing.",
        "Great phone, but too expensive.",
        "This is the best phone I've ever used.",
        "The performance is very laggy and slow.",
        "Not worth the money at all.",
        "The screen resolution is mind-blowing!",
        "Terrible customer service experience."
    ]
}

df = pd.DataFrame(data)

# Step 2: Preprocess the Data
nltk.download('stopwords')  # Download the stopwords from NLTK
stop_words = set(stopwords.words('english'))

# Function to clean tweet text
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+", "", tweet)  # Remove URLs
    tweet = re.sub(r"@\w+", "", tweet)  # Remove mentions
    tweet = re.sub(r"#\w+", "", tweet)  # Remove hashtags
    tweet = re.sub(r"[^\w\s]", "", tweet)  # Remove special characters
    tweet = tweet.lower()  # Convert to lowercase
    tweet = ' '.join([word for word in tweet.split() if word not in stop_words])  # Remove stopwords
    return tweet

# Apply the cleaning function to the tweets
df['cleaned_tweet'] = df['tweet'].apply(clean_tweet)

# Step 3: Perform Sentiment Analysis
# Function to get sentiment
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Apply sentiment analysis
df['sentiment'] = df['cleaned_tweet'].apply(get_sentiment)

# Display the dataframe with sentiments
print(df[['tweet', 'cleaned_tweet', 'sentiment']])

# Step 4: Visualize Sentiment Patterns

# Set the theme for the plots
sns.set_theme(style="whitegrid")

# Plot the distribution of sentiments
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.show()

# Example of a pie chart
plt.figure(figsize=(6, 6))
df['sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightcoral', 'gold', 'yellowgreen'])
plt.title('Sentiment Distribution')
plt.ylabel('')
plt.show()

# Optional Step: Time Series Analysis (if you have a 'date' column)
# For this example, we'll simulate a date column

# Simulate a date column (Assuming data collected over 10 days)
import numpy as np
df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')

# Group by date and sentiment
sentiment_over_time = df.groupby(['date', 'sentiment']).size().unstack().fillna(0)

# Plot sentiment over time
sentiment_over_time.plot(kind='line', figsize=(10, 6))
plt.title('Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.show()
