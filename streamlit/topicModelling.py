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

st.set_page_config(layout="wide")
#num_topics = 5
vizFile = 'LDA_viz.html'
wordcloudName= 'wordcloud.png'

# Function to filter the DataFrame based on date range and star rating
def filter_reviews(data, start_date, end_date, star_rating):
    filtered_data = data[(data['dateExp'] >= start_date) & (data['dateExp'] <= end_date) & (data['starRating'].isin(star_rating))]
    return filtered_data

# Function to display reviews in a scrollable window
#def display_reviews(reviews):
    #st.text_area("Reviews", "\n\n".join(reviews['reviews']), height=300)

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
    #bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
    corpus_vecs = [dictionary.doc2bow(doc) for doc in corpus]
    
    # create tf-idf model and convert the bow vector to tfidf vectors
    #tfidf_model = models.TfidfModel(bow_corpus)
    #tfidf_corpus = tfidf_model[bow_corpus]
    
    return dictionary, corpus_vecs

def lemmatize(tokenizedText):
    lemmatizer = WordNetLemmatizer()
    filteredTokens = [lemmatizer.lemmatize(token) for token in tokenizedText]
    return filteredTokens

def train_lda_model(corpus, num_topics, dictionary):
    # Train lda model on tf-idf corpus
    lda_model = models.LdaModel(corpus=corpus,
                                num_topics=num_topics,
                                id2word=dictionary,
                                passes=30)
    
    return lda_model


def generate_wordcloud(corpus_bigrams, wordcloudName):
    long_string = ' '.join(['_'.join(bigram.split(' ')) for doc_bigrams in corpus_bigrams for bigram in doc_bigrams])
    wordcloud = WordCloud(scale = 3,
        background_color='white',
        max_words=200,
        max_font_size=50,
        colormap='BrBG',
        random_state=42
                          
    ).generate(long_string)
    
    wordcloud.to_file(wordcloudName)
    

def visualizeLDA(lda_model_gensim, corpus_vecs, dictionary, fileName):
    vis_data = gensimvis.prepare(lda_model_gensim, corpus_vecs, dictionary)
    pyLDAvis.save_html(vis_data, fileName)
    
def main():
    
    # Set app title
    st.title('Topic Analysis for Customer Reviews')
    
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
    
    
    # Resize the image
    image = Image.open(wordcloudName)
    resized_image = image.resize((1000, 500))

    # Display resized image
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
