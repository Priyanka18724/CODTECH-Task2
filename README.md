# CODTECH-Task2
import tweepy
import pandas as pd
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
query = 'data science'  
tweets = tweepy.Cursor(api.search, q=query, lang='en', tweet_mode='extended').items(100)
data = pd.DataFrame(data=[tweet.full_text for tweet in tweets], columns=['text'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
def preprocess_text(text):
    text = text.lower() 
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'[^a-z\s]', '', text)  
    tokens = word_tokenize(text)  
    tokens = [word for word in tokens if word not in stopwords.words('english')]  
    return ' '.join(tokens)
data['clean_text'] = data['text'].apply(preprocess_text)
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity
data['sentiment'] = data['clean_text'].apply(get_sentiment)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.histplot(data['sentiment'], bins=20, kde=True)
plt.title('Sentiment Score Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

