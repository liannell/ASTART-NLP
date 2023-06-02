# Streamlit - Topic modelling


import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re
import nltk
nltk.download("all")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
import gensim
from gensim.utils import simple_preprocess 
from nltk import bigrams
from gensim import corpora, models
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis
import webbrowser
from PIL import Image
import random
from transformers import pipeline

st.set_page_config(layout="wide")
#num_topics = 5
vizFile = 'LDA_viz.html'
wordcloudName= 'wordcloud.png'

# Create the summarizer pipeline
summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

# Function to filter the DataFrame based on date range and star rating
def filter_reviews(data, start_date, end_date, star_rating):
    filtered_data = data[(data['dateExp'] >= start_date) & (data['dateExp'] <= end_date) & (data['starRating'].isin(star_rating))]
    return filtered_data

# Function to display reviews in a scrollable window
def display_reviews(reviews, keyword):
    if keyword:
        # Filter the DataFrame based on the keyword
        filtered_reviews = reviews[reviews['reviews'].str.contains(keyword, case=False)]
        
        # Display the filtered reviews in a scrollable window
        st.subheader("Filtered Reviews")
        st.text_area("Reviews", "\n\n".join(filtered_reviews['reviews']), height=300)
    else:
        # Display the unfiltered reviews in a scrollable window
        st.subheader("All Reviews")
        st.text_area("Reviews", "\n\n".join(reviews['reviews']), height=300)
    
    
    
# Remove Stopwords
stopwords  = stopwords.words('english')
def removeStopwords(tokenizedText):
    filteredTokens = [token for token in tokenizedText if token.lower() not in stopwords]
    return filteredTokens

# Preprocess: lowercase, tokenization, filtering
def preprocess_corpus(data):
    processed_corpus = data.apply(lambda x: simple_preprocess(x))
    return processed_corpus

# Bigrams and Vectors
def create_bigrams(corpus):
    # Create a list to hold the bigram models
    corpus_bigrams = []

    # Create bigrams for each document in the corpus
    for doc in corpus:
        doc_bigrams = list(bigrams(doc))
        doc_bigrams = [' '.join(bigram) for bigram in doc_bigrams]  # Convert bigrams to strings
        corpus_bigrams.append(doc_bigrams)

    return corpus_bigrams

def create_dict_tfidf(corpus):
    # create dict using the preprocessed corpus
    dictionary = corpora.Dictionary(corpus)
    
    # create bag-of-words representation of the corpus
    bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    
    # create tf-idf model and convert the bow vector to tfidf vectors
    tfidf_model = models.TfidfModel(bow_corpus)
    tfidf_corpus = tfidf_model[bow_corpus]
    
    corpus_vecs = tfidf_corpus
    
    return dictionary, corpus_vecs

# Lemmatization
def lemmatize(tokenizedText):
    lemmatizer = WordNetLemmatizer()
    filteredTokens = [lemmatizer.lemmatize(token) for token in tokenizedText]
    return filteredTokens

# LDA Model
def train_lda_model(corpus, num_topics, dictionary):
    # Train lda model on tf-idf corpus
    lda_model = models.LdaModel(corpus=corpus,
                                num_topics=num_topics,
                                id2word=dictionary,
                                passes=20)
    
    return lda_model


# WordCloud
def generate_wordcloud(corpus_bigrams, wordcloudName):
    long_string = ' '.join(['_'.join(bigram.split(' ')) for doc_bigrams in corpus_bigrams for bigram in doc_bigrams])
    wordcloud = WordCloud(scale = 3,
        background_color='white',
        max_words=300,
        max_font_size=35,
        colormap='BrBG',
        random_state=42
                          
    ).generate(long_string)
    
    wordcloud.to_file(wordcloudName)
    
# PyLDAvis
def visualizeLDA(lda_model_gensim, corpus_vecs, dictionary, fileName):
    vis_data = gensimvis.prepare(lda_model_gensim, corpus_vecs, dictionary)
    pyLDAvis.save_html(vis_data, fileName)
    
def generate_random_summary(corpus, length_limit):
    if len(corpus) == 0:
        return "Corpus is empty."

    #article = ' '.join(corpus.apply(lambda x: ' '.join(x)))
    #article = ' '.join(' '.join(x) for x in corpus)
    
    flattened_corpus = []
    for doc in corpus:
        if isinstance(doc, (list, tuple)):
            flattened_corpus.extend(doc)
        elif isinstance(doc, (float, int, str)):
            flattened_corpus.append(str(doc))
        else:
            print(f"Unsupported data type: {type(doc)}")
    
    article = ' '.join(flattened_corpus)
    article_length = len(article)

    if article_length <= length_limit:
        start_idx = 0
        end_idx = article_length
    else:
        start_idx = random.randint(0, article_length - length_limit)
        end_idx = start_idx + length_limit
    
    article_chunk = article[start_idx:end_idx]
    
    # Generate the summary using the pre-initialized summarizer pipeline
    summary = summarizer(article_chunk, max_length=1000, min_length=50, do_sample=False)
    
    return summary[0]['summary_text']

# Function to generate random summaries for each topic
def generate_random_topic_summaries(topic_dataframes, length_limit):
    topic_summaries = {}

    for topic, dataframe in topic_dataframes.items():
        reviews = dataframe['reviews']
        random_summary = generate_random_summary(reviews, length_limit)
        topic_summaries[topic] = random_summary

    return topic_summaries


def generate_topics_df(lda_model, corpus_vecs, num_topics):
    # Assign topics to documents
    document_topics = []
    for i, doc in enumerate(corpus_vecs):
        doc_topics = lda_model.get_document_topics(doc)
        document_topics.append([prob for _, prob in doc_topics])

    # Convert document_topics into a DataFrame
    topics_df = pd.DataFrame(document_topics)

    # Rename the columns to represent topics
    topics_df.columns = [f"Topic{i+1}" for i in range(num_topics)]
    
    return topics_df



### Main Execution    
def main():
    
    # Set app title
    st.title('Automated Summary and Topic Analysis for Customer Reviews')
    
    PATH = os.getcwd()

    filesArray = os.listdir(f'{PATH}/archive')
    csvFiles = [file for file in filesArray if file.endswith(".csv")]
    csvFiles.sort()
        
    # Combine into one data frame
    data = []
    for i in csvFiles:
        data.append(pd.read_csv(f'{PATH}/archive/{i}'))

    reviewsRaw = pd.concat(data, ignore_index=True)

    columnNames = {'Date of Exp' : 'dateExp', 'Star Rating' : 'starRating', 'Reviews': 'reviews'}
    reviewsRaw = reviewsRaw.rename(columns=columnNames)
    reviewsRaw['dateExp'] = pd.to_datetime(reviewsRaw['dateExp'], format = 'mixed')
    
    
    # Set default values for start and end dates
    default_start_date = datetime(2023, 1, 1)
    default_end_date = datetime(2023, 5, 26)

    # Create date input components
    start_date = st.date_input("Start Date", value=default_start_date)
    end_date = st.date_input("End Date", value=default_end_date)
    
    # Define the valid date range
    valid_start_date = pd.to_datetime('2017-01-01').date()
    valid_end_date = pd.to_datetime('2023-05-26').date()
    
    
    # Validate the selected date range
    if start_date > end_date:
        st.error("Error: Start date must be before or equal to end date.")
        return

    if start_date < valid_start_date or end_date > valid_end_date:
        st.error(f"Error: Date range must be between {valid_start_date} and {valid_end_date}.")
        return
    
    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Add star rating selection
    star_rating = st.multiselect("Select star rating", [1, 2, 3, 4, 5])
    
    # Add number of topics selection
    num_topics = st.slider("Select the number of topics", min_value=2, max_value=10, value=5)
    
    # Filter the DataFrame based on date range and star rating
    reviews = filter_reviews(reviewsRaw, start_date, end_date, star_rating)

    #display_reviews(reviews)
    
    # Data Preprocessing
    content = (reviews['reviews'])
    corpus = []
    preprocessed_corpus = preprocess_corpus(content)
    
    preprocessed_corpus = preprocess_corpus(content)
    preprocessed_corpus = preprocessed_corpus.apply(removeStopwords)
    preprocessed_corpus = preprocessed_corpus.apply(lemmatize)
    
    # Create the bigrams in the corpus
    corpus_bigrams = create_bigrams(preprocessed_corpus)

    # Generate wordcloud
    generate_wordcloud(preprocessed_corpus, wordcloudName)
    
    # Create the dictionary and TF-IDF corpus with bigrams
    dictionary, corpus_vecs = create_dict_tfidf(corpus_bigrams)
    
    # Train LDA MODEL
    lda_model_gensim = train_lda_model(corpus_vecs, num_topics, dictionary)
 
    # Visualize pyLDAvis
    visualizeLDA(lda_model_gensim, corpus_vecs, dictionary, vizFile)
    
    
    # Assign topics to reviews
    topics_df = generate_topics_df(lda_model_gensim, corpus_vecs, num_topics)
    # Consolidate
    processedReviews = preprocessed_corpus.rename('processedReviews')
    reviewsNtopics = pd.concat([reviews['reviews'], processedReviews, topics_df], axis=1)
    
    # Generate a random summary for each topic on button click
    
    threshold = 1 / num_topics
    topic_dataframes = {}

    for column in topics_df.columns:
        topic_reviews = reviewsNtopics[reviewsNtopics[column] >= threshold]  # Filter reviews with score greater than or equal to threshold
        topic_dataframes[column] = topic_reviews
    
    # Button
    # Generate a random summary for each topic on button click
    if st.button("Generate Random Summaries for each topic"):
        length_limit = 1000  # Adjust the length limit as needed
        topic_summaries = generate_random_topic_summaries(topic_dataframes, length_limit)

        # Display the random summaries
        st.subheader("Random Summaries")
        for topic, summary in topic_summaries.items():
            st.write(f"Summary for {topic}: {summary}")
    
    # Wordcloud
    image = Image.open(wordcloudName)
    resized_image = image.resize((1000, 500))
    st.subheader("Wordcloud")
    st.image(resized_image, use_column_width=True)


    # Display visualization from pyLDAvis with adjusted zoom level
    st.subheader("Topics and Keywords")
    html_file = open(vizFile, 'r', encoding='utf-8').read()

    # Wrap the HTML content in a div and apply CSS styles for zooming
    zoom_level = 0.8
    html_content = f"<div style='zoom: {zoom_level};'>{html_file}</div>"
    # Display the HTML content
    st.components.v1.html(html_file, width=1600, height=800)

     # Filter the DataFrame based on the keyword
    keyword = st.text_input("Enter a keyword")
    display_reviews(reviews, keyword)


if __name__ == '__main__':
    main()
