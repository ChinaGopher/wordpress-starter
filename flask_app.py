
import pandas as pd
import numpy as np
import re
import os
import PyPDF2
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from flask import render_template, Flask, request
import flask
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import KMeans


stemmer=SnowballStemmer("english")
stop = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')
from random import shuffle

def pdf_to_text(pdf):

    pdfReader = PyPDF2.PdfFileReader(pdf)
    numPage=pdfReader.numPages
    pdftext=" "
    for page in range(numPage):
        pageObj =pdfReader.getPage(page)
        text= pageObj.extractText()
        pagetext="".join(text)
        pdftext=" ".join([pagetext, pdftext])
    

    all_words= re.findall("[a-zA-Z]+", pdftext)
    affterstemmer=[]
    for word in all_words:
        affterstemmer.append(stemmer.stem(word))
    afterstop=[]
    for word in affterstemmer:
        if word not in stop:
            afterstop.append(word)
    if len(afterstop)>50:
        afterstop.reverse()
        
        return " ".join(afterstop)


THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_FOLDER)

## Geting ready the training data
data=pd.read_csv('readydata.csv')

count_vect = CountVectorizer(lowercase = False, max_df = .6)
tfidf_transformer = TfidfTransformer()
X_train_counts = count_vect.fit_transform(data['0'])
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# K-means model
clf = KMeans(n_clusters=2,  max_iter=100 , random_state=0).fit(X_train_tfidf)
def getPredictions(clf,count_vect,tfidf_transformer,X_test):