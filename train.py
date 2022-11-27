# Import the necessary packages
import ast
import pandas as pd
import pymysql
import stanfordnlp
import numpy as np
import sys
import spacy
import pickle
import re
import gensim
import en_core_web_sm
import os
import json
import joblib
import neuralcoref
import pandas as pd
import nltk
import os
import re
import warnings
import time

warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from spacy.symbols import nsubj, VERB
from gensim import corpora
from nltk import flatten
from pandas.io.json import json_normalize
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import wordnet as wn
from sqlalchemy import create_engine
from datetime import datetime
from textblob import TextBlob
import xlrd

# stanfordnlp.download('en')
now = datetime.now()
#nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)
# stf = stanfordnlp.Pipeline(processors='tokenize,lemma,pos,depparse', treebank='en_ewt', use_gpu=True,
#                            pos_batch_size=3000)  # This sets up a default neural pipeline in English

import re
from nltk.corpus import stopwords

english_stop_words = stopwords.words('english')
REPLACE_NO_SPACE = re.compile(r"(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile(r"(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "
conj_list = ['for', 'or','and','nor','but','yet','so']

def load_file_for_training_classifier(file):
    if len(file) == 0:
        print("Do provide a data file or json file")
    # Converting the  file into a dataframe
    try:
        if file :  # data file
            if (file.endswith('.xlx') or file.endswith('.xlsx')):
                df_input = pd.read_excel(file)
            elif file.endswith('.csv'):
                df_input = pd.read_csv(file)
            elif (file.endswith('.txt') or file.endswith('.json')):
                with open(file) as json_file:
                    data = json.load(json_file)
                    df_input = pd.DataFrame.from_dict(data, orient='columns')

        df_input = df_input.drop(df_input.index[df_input['Rating'] == 'Rating'])
        df_input.astype({'Rating': 'float'}).dtypes
        if 'RepuGen Review' in df_input.columns:
            df_input = df_input.rename(columns={"RepuGen Review": "Comment"})
        elif 'Review' in df_input.columns:
            df_input = df_input.rename(columns={"Review": "Comment"})
        return df_input

    except Exception as e:
        return "error"



def data_preprocessing(comment):
    removed_stop_words = []

    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in comment]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in comment]

    for review in reviews:
        removed_stop_words.append(
            ' '.join([word for word in review.split()
                      if word not in english_stop_words])
        )

    return removed_stop_words


# Converting the data into three classes which keeps the classifier at optimal performance
def classfying_train_lables(df):
    df.loc[df['Rating'] < 5, 'Rating'] = 0
    df.loc[(df['Rating'] > 4) & (df['Rating'] < 8), 'Rating'] = 1
    df.loc[df['Rating'] > 7, 'Rating'] = 2
    df['Rating'] = df['Rating'].astype(str)

    df['Rating'] = df['Rating'].replace(str(0), 'Negative')
    df['Rating'] = df['Rating'].replace(str(1), 'Neutral')
    df['Rating'] = df['Rating'].replace(str(2), 'Positive')
    return df['Rating']


def train_classifier(file):
    df = load_file_for_training_classifier(file)
    if "error" in df:
        return {"Error:":"File format incorrect"}

    df['Comment'] = df['Comment'].astype(str)
    df['Comment'] = data_preprocessing(df['Comment'])
    df['Rating'] = classfying_train_lables(df)

    # ncount vectorizer - vectorizes the words into bigrams
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(2, 2))
    ngram_vectorizer.fit(df['Comment'])
    vectorizer_filename = 'vectorizer' + str(now) + '.pkl'
    #vectorizer_filename = 'vectorizer' + '.pkl'
    joblib.dump(ngram_vectorizer, './upload/' + vectorizer_filename)

    X = ngram_vectorizer.transform(df['Comment'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, df['Rating'], train_size=0.8)

    maximum_acc = 0
    hyper_parameter = 0
    # finding the best hyper parameter
    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        svm = LinearSVC(C=c)
        svm.fit(X_train, y_train)

        # svm_bigram_model - uses the vectorised data to classify the data.
        filename = 'classifier_model' + str(now) + '.pkl'
        #filename = 'classifier_model' + '.pkl'

        if accuracy_score(y_test, svm.predict(X_test)) > maximum_acc:
            maximum_acc = accuracy_score(y_test, svm.predict(X_test))
            hyper_parameter = c

    y_predict = svm.predict(X_test)
    final_count_ngram = LinearSVC(C=hyper_parameter)
    final_count_ngram.fit(X, df['Rating'])
    # print('Max Accuracy is', accuracy_score(y_test, final_count_ngram.predict(X_test)))
    score_dict = {'Vectorizer_filename': vectorizer_filename, 'Classifier_model_name': filename,
                  'Max Accuracy is': accuracy_score(y_test, final_count_ngram.predict(X_test))}
    joblib.dump(svm, './upload/' + filename)
    return score_dict

if __name__ == "__main__":
    file = sys.argv[1]
    print(file)
    #file = "C:/Users/Asus/Downloads/Microsoft.SkypeApp_kzf8qxf38zg5c!App/All/First10000.xlsx"
    #file ="c"
    var = train_classifier(file)
    var1 = json.dumps(var)
    print("output: ", var1)

