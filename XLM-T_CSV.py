import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load the CSV file into a DataFrame
df = pd.read_csv('D:\KULIAH\Semester 5\Akademik\ETI\Week 3\skripsi pak mentri\pak-mentri-skripsi.csv')  

# Create empty lists to store sentiment analysis results
sentiments = []
positive_probs = []
neutral_probs = []
negative_probs = []

# Load model and tokenizer
roberta = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# Iterate over the rows of the DataFrame
for tweet_text in df['full_text']:
    # Preprocess tweet
    tweet_words = []

    for word in tweet_text.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)

    tweet_proc = " ".join(tweet_words)

    # Sentiment analysis
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Extract sentiment probabilities
    negative_prob, neutral_prob, positive_prob = scores.tolist()

    # Determine sentiment label
    sentiment_label = labels[scores.argmax()]

    # Append results to lists
    sentiments.append(sentiment_label)
    positive_probs.append(positive_prob)
    neutral_probs.append(neutral_prob)
    negative_probs.append(negative_prob)

# Add new columns to the DataFrame for sentiment analysis results
df['Sentiment'] = sentiments
df['Positive_Prob'] = positive_probs
df['Neutral_Prob'] = neutral_probs
df['Negative_Prob'] = negative_probs

# Save the modified DataFrame to a new CSV file
df.to_csv('D:\KULIAH\Semester 5\Akademik\ETI\Week 3\skripsi pak mentri\-output-pak-mentri-skripsi.csv', index=False)
