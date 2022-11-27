# Import the necessary packages
import ast
import pandas as pd
# import pymysql
import numpy as np
import spacy
import pickle
import re
# import gensim
# import en_core_web_sm
import os
import json
import joblib
import neuralcoref
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
# import xlrd

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
negative_stop_words = ["no", "nor", "not", "don't", "aren't", "couldn't","didn't","doesn't","hadn't","hasn't","haven't","mustn't","shouldn't","wouldn't","isn't","wasn't","weren't"]
    
REPLACE_NO_SPACE = re.compile(r"(\.)|(\;)|(\:)|(\!)|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile(r"(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "
conj_list = ['for', 'or','and','nor','but','yet','so']

DEFAULT_THRESHOLD_CONST = 0.51
DEFAULT_NEG_THRESHOLD_CONST = 0.4852

# karthick aravidan codes Start
# -----------------------------------------------
# df1 = pd.read_csv("./Dataset/hidden_entity_list.csv")
# list_of_entity_name = df1['name'].to_list()

def round_fload_val(val):
    return round(val, 2)

def new_neural_coref(name, text):
    text = (text.lower()).strip()
    text = " "+text
    pronoun_list = [' he ', ' she ', ' her ', ' his ', ' your ', ' you ', ' him ', ' her ']
    #name filter process
    name = name.lower()
    name_arr = name.split(", ")
    alter_name = check_spelling(name_arr[0])
    alter_name = str(alter_name)
    alter_name = ' '+alter_name+' '


    first_occurrence_index = len(text)
    total_cnt = len(text)
    first_occurrence_pronoun = ''
    for val in pronoun_list:
        index_val = text.find(val)
        if (index_val != -1) and index_val < first_occurrence_index:
            first_occurrence_index = index_val
            first_occurrence_pronoun = val
    
    if first_occurrence_index > -1 and (first_occurrence_index != total_cnt):
        text = text.replace(first_occurrence_pronoun,alter_name,1)
    text = text.strip()
    text = text.replace(".", " . ")
    doc = nlp(text.lower())
    return doc._.coref_resolved


def neural_coref(name, text):
    text = text.split(' ')
    if text[0].lower() in ['he', 'she', 'her', 'his', 'your', 'you', 'him', 'her']:
        text[0] = name
    text = ' '.join(text)
    text = text.replace(".", " . ")
    doc = nlp(text.lower())
    return doc._.coref_resolved


def check_spelling(text):
    text = text.replace("dr.", "dr ")
    text = text.replace("dr .", "dr ")
    #text = text.replace("dr", "")
    text = text.replace(",", "")
    text = TextBlob(text)
    return text  # .correct()

def name_normalizer(text):
    text = text.replace("dr.", "")
    text = text.replace("dr .", "")
    text = text.replace("dr", "")
    text = text.replace(",", "")
    text = TextBlob(text)
    return text  # .correct()


def get_removal_conjugation(blob):
    format_pos = ['PROPN', 'CCONJ', 'NOUN']
    final_sentence = ""
    error = []
    for sentence in blob.sentences:
        doc = nlp(str(sentence))
        flag = False
        pos = [token.pos_ for token in doc]
        if len(pos) >= 3:
            test_sentence = str(sentence).split(" ")
            index = [n for n, x in enumerate(pos) if x == 'CCONJ']
            for i in index:
                try:
                    list_noun = ['PROPN', 'NOUN', 'PRON']
                    if pos[i - 1] in list_noun:
                        flag = True
                        after_cconj = pos
                    else:
                        after_cconj = pos[i + 1:]
                    if len(after_cconj) >= 4:
                        if 'CCONJ' not in after_cconj[1:4]:
                            next_three_pos = pos[i + 1:i + 4]
                            if 'NOUN' or 'PROPN' in list(set(list_noun) & set(next_three_pos)):

                                print("*******")
                                if 'PROPN' in next_three_pos:
                                    conj_index = next_three_pos.index('PROPN')
                                if 'NOUN' in next_three_pos:
                                    conj_index = next_three_pos.index('NOUN')
                                if 'PRON' in next_three_pos:
                                    conj_index = next_three_pos.index('PRON')
                                #                                     print(next_three_pos.index('NOUN'))
                                #                                 print(pos[i+conj_index:i+conj_index+4])
                                if ('AUX' in pos[i + conj_index:i + conj_index + 4] or 'VERB' in pos[
                                                                                                 i + conj_index:i + conj_index + 4]) and flag == False:
                                    test_sentence[i] = "."
                #                                     print("Condition 3 is passed")
                except Exception as e:
                    error.append(sentence)
            sentence = " ".join(test_sentence)
        final_sentence += str(sentence) + " "

    return final_sentence, error


def get_output(name, comment):
    neural_coref_output = neural_coref(name, comment)

    blob = check_spelling(neural_coref_output)

    result, error = get_removal_conjugation(blob)
    print(result)
    return result, error

#This nlp_core_dataProces method do the pronoun resoultion and specll check all nlp process
def nlp_core_dataProces(name, comment):
    neural_coref_output = new_neural_coref(name, comment)
    text = remove_specialchars(neural_coref_output)
    text = " ".join(text.split()) 
    text = text.replace("'", "`")
    blob = TextBlob(text)
    split_sentences = split_sentence_simplified(blob)
    return split_sentences


# --------------------------------------------------------------


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


## Preparing data to Train the Classifier:

# Loading the file to train the classifier

def load_file_for_training_classifier(file):
    if len(file) == 0:
        print("Do provide a data file or json file")
    # Converting the  file into a dataframe
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


# The classifier gets trained here and the model gets saved as a pickle file.

def train_classifier(file):
    df = load_file_for_training_classifier(file)

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


### Extracting hidden entities and sentences associated with them and predicting their sentiment
## along with predicting the sentiment of the entire review. Note: Rating, Date Columns are no longer required.

# Loading the file to extract hidden entities. - Step-1

def load_file_to_extract_hidden_entities(file, vectorizer_filename, classifier_filename, data_json):
    # print(flag, "flag")
    if len(file) == 0 and len(data_json) == 0:
        print("Do provide a data file or json file")
        # Needs a check
        return "Do provide a data file or json file"

    if file and len(data_json) == 0:
        if (file.endswith('.xlx') or file.endswith('.xlsx')):
            df_extract_entities = pd.read_excel(file)
            # print("opened with excel")
        elif file.endswith('.csv'):
            df_extract_entities = pd.read_csv(file)
        elif (file.endswith('.txt') or file.endswith('.json')):
            with open(file) as json_file:
                data = json.load(json_file)
                # print("opened")
                df_extract_entities = pd.DataFrame.from_dict(data, orient='columns')
    if len(data_json) > 0:
        data_json = json.loads(data_json)
        # print(data_json)
        # print("****Load****")
        df_extract_entities = json_normalize(data_json)

        # pd.DataFrame.from_records(data_json)
        # print(df_extract_entities.head())
    # df_extract_entities['Date of Comment'] =df_extract_entities['Date of Comment'].astype(str)
    # df_extract_entities = df_extract_entities.drop(df_extract_entities.index[df_extract_entities['Rating']=='Rating'])
    # df_extract_entities.astype({'Rating':'float'}).dtypes
    if 'RepuGen Review' in df_extract_entities.columns:
        df_extract_entities = df_extract_entities.rename(columns={"RepuGen Review": "Comment"})
    elif 'Review' in df_extract_entities.columns:
        df_extract_entities = df_extract_entities.rename(columns={"Review": "Comment"})
    df_extract_entities = df_extract_entities.rename(columns={"Provider": "Name"})
    df_extract_entities['Comment'] = df_extract_entities['Comment'].astype(str)

    ngram_vectorizer = joblib.load(vectorizer_filename)
    svm = joblib.load(classifier_filename)
    # print("svm")
    return df_extract_entities, ngram_vectorizer, svm


def simple_sentences_breakdown(name, comment):
    # doc= stf(comment)
    simple_sentence_list_final, error = get_output(name, comment)
    #  removal_conjugation(doc)
    print(error)
    return simple_sentence_list_final


# The trained model reads the data and does the preprocessing of stop words removal and vectorisation
# and then classifies the data accordingly.


def classifying_the_sentence(comment, ngram_vectorizer, svm):
    removed_stop_words = []
    pre_rev = [REPLACE_NO_SPACE.sub(NO_SPACE, str(comment))]
    pre_rev = [REPLACE_WITH_SPACE.sub(SPACE, str(comment))]
    ngram_vectorizer = ngram_vectorizer
    svm = svm
    filltered_stop_words = [ elem for elem in english_stop_words if elem not in negative_stop_words] 
    for review in pre_rev:
        removed_stop_words.append(
            ' '.join([word for word in review.split()
                      if word not in filltered_stop_words])
        )

    predict_data = removed_stop_words
    # ngram_vectorizer = joblib.load(vectorizer_model)
    x = ngram_vectorizer.transform(predict_data)
    # svm = joblib.load(classifier_model)
    # round off the values two decimal places
    model_prediction_probability = svm._predict_proba_lr(x)
    arr_ele = []
    for val in model_prediction_probability[0]:
        arr_ele.append(round_fload_val(val))
    mat_arr = [arr_ele]
    rounded_model_prediction_probability = np.array(mat_arr)
    model_prediction = svm.predict(x)
    # print("Comment-",comment)
    # print("rounded_model_prediction_probability-",rounded_model_prediction_probability)
    # print("model_prediction-",model_prediction)
    return rounded_model_prediction_probability, model_prediction


#Pronoun resolution is the process of identifying the first pronoun and replacing it with the entity name
def pronoun_resolution(row):
    breakdown_list = simple_sentences_breakdown(row['Name'], row['Comment'])
    print(breakdown_list)
    return [breakdown_list]


#Extracting the hidden entities from the data.

def extract_hidden_entities(review, list_of_entity_name):

    #print("after spliting:"+review)
    if type(list_of_entity_name) != list:
        list_of_entity_name = ast.literal_eval(list_of_entity_name)
    all_small_letter = []
    for i_entity in list_of_entity_name:
        all_small_letter.append(i_entity.lower())
    list_of_entity_name = all_small_letter
    extracted_sentences = {}
    review_without_hidden_entity = []
    sentence_count = 0
    acceptable_dep = ['pobj', 'dobj', 'iobj', 'ROOT', 'conj', 'npadvmod', 'nmod', 'attr', 'dep']
    for sentence in sent_tokenize(review):
      #  print(sentence)
        sentence_count += 1
        doc = nlp(sentence.strip())
        i = 0
        for token in doc:
            i = i+1
            word = ''
            #print(token, token.dep_)
            if str(token.dep_) == 'nsubj' or str(token.dep_) == 'nsubjpass' or (( ((token.pos_)=='NOUN') or ((token.pos_)=='PROPN') or ((token.pos_)=='PRON')) and ((token.dep_) in acceptable_dep)) or (token.text.lower() == 'dr'):
                # print("^^^^^^^^^^^^^^^^")
                token_index = token.i
                if token_index > 0:
                    prev_token = doc[token_index - 1]
                    combined_token = prev_token.text.lower() + " " + token.text.lower()
                    if combined_token in list_of_entity_name:
                        extracted_sentences[combined_token] = sentence
                    elif (prev_token.dep_ == 'compound' or prev_token.dep_ == 'poss') and (prev_token.text.lower() in list_of_entity_name) and (token.text.lower() in list_of_entity_name):
                        word = prev_token.text.lower() + " " + token.text.lower()
                        extracted_sentences[word] = sentence
                    elif (prev_token.dep_ == 'compound' or prev_token.dep_ == 'poss') and (prev_token.text.lower() in list_of_entity_name):
                        word = prev_token.text.lower()
                        if (word == 'dr') and token.dep_ == 'nsubj':
                            word = word + ' ' + token.text.lower()
                        extracted_sentences[word] = sentence
                    elif (token.text.lower() in list_of_entity_name):
                        word = token.text.lower()
                        if (word == 'dr') and (i != len(doc)) and (token.pos_ == 'PROPN'):
                            next_word = doc[token_index + 1]
                            word = word + ' ' + next_word.text.lower()
                            if word not in extracted_sentences.keys():
                                extracted_sentences[word] = sentence
                        else:
                            extracted_sentences[word] = sentence
                elif(token.text.lower() in list_of_entity_name):
                    word = token.text.lower()
                    if (word == 'dr') and (i != len(doc)) and (token.pos_ == 'PROPN'):
                        next_word = doc[token_index + 1]
                        word = word + ' ' + next_word.text.lower()
                        if word not in extracted_sentences.keys():
                            extracted_sentences[word] = sentence
                    else:
                        extracted_sentences[word] = sentence

                # Extracting the entities only if they belong to this list of pre-defined words
                # KC: 11/6/20: The reason we extract 2 entities like office, staff if the input review has 'office staff' is
                # because of the prev_token logic -> validate this by commenting that prev_token and run.
                # Old pipeline:
                '''
                if token.text.lower() in list_of_entity_name:  # ['technicians','facility','assistant','desk','rep','lpn','office','rn','establishment','provider','specialists','employee','nurse','np','receptionist','pa','rn','staff','attendant','practitioner']:
                    extracted_sentences[token.text] = sentence
                if prev_token.text.lower() in list_of_entity_name:  # ['technicians','facility','assistant','desk','rep','lpn','office','rn','establishment','provider','specialists','employee','nurse','np','receptionist','pa','rn','staff','attendant','practitioner']:
                    extracted_sentences[prev_token.text] = sentence
                '''
        # named entity recognition
        for ent in doc.ents:
            pos_list = []
            named_text = ent.text.strip()
            named_text = named_text.capitalize()
            parse_ent = nlp(named_text)
            for tkn in parse_ent:
                pos_list.append(tkn.pos_)
            if 'ADJ' not in pos_list:
                new_sentence = sentence.split(' ')
                name_index = new_sentence.index(str(ent)) if ((str(ent)) in new_sentence) else 0
                key_ent = str(ent)
                if (name_index > 0) and (new_sentence[name_index-1] == 'dr'):
                    key_ent = new_sentence[name_index-1] +' '+ key_ent
                extracted_sentences[key_ent] = sentence
    # if sentence_count>1:
    # print(extracted_sentences)
    # print("*****")
    # print(len(extracted_sentences))

    # print("(*)" * 10)
    if len(extracted_sentences) == 0:
        # print(review)
        review_without_hidden_entity.append(review)
        # print(' Extracted without hidden entity', review_without_hidden_entity)
    else:
        for i in extracted_sentences.values():
            # print(i)
            # print("*-" * 10)

            review = review.replace(i, "")
            review = review.strip()
        if review != []:
            review_without_hidden_entity.append(review)
        # print(' Extracted without hidden entity not empty', review_without_hidden_entity)

    review_without_hidden_entity = ''.join(review_without_hidden_entity)
    return extracted_sentences, review_without_hidden_entity

def makeRealValue(val):
    c = val
    n = c * -10
    x = 10 - n
    y = x * 0.15
    z = 1 - y
    c = c * -1
    t = c - z
    return(round(t, 2))


def get_sentiment_score_textblob(sentence, svm_pos_score, max_threshold=DEFAULT_THRESHOLD_CONST, mim_threshold=DEFAULT_NEG_THRESHOLD_CONST):
    max_threshold = float(max_threshold)
    sentence = sentence.strip()
    if len(sentence) != 0:
        zen = TextBlob(sentence)
        text_blob_score = float(zen.sentences[0].polarity)
        getRealVal = makeRealValue(text_blob_score)
        avg_score = (getRealVal + float(svm_pos_score))/2
        avg_label = ''
        if avg_score >= max_threshold:
            avg_label = 'Positive'
        elif avg_score >= mim_threshold and avg_score < max_threshold:
            avg_label = 'Neutral'
        else:
            avg_label = 'Negative'
        return {"score": round_fload_val(avg_score), "sentiment": avg_label}
    else:

        return "sentence is zero"


def get_text_blob_score(text):
    zen = TextBlob(text)
    #     print(zen.sentences[0].polarity)
    for sentence in zen.sentences:
        # print("Word: ",sentence)
        if sentence.sentiment.polarity > 0:
            score = sentence.sentiment.polarity
            sentiment = "positive"
        elif sentence.sentiment.polarity == 0:
            score = sentence.sentiment.polarity
            sentiment = "neutral"
        else:
            score = sentence.sentiment.polarity
            sentiment = "negative"
    # print(text)
    # print({"word": text, "score": score, "sentiment": sentiment})
    return {"word": text, "score": score, "sentiment": sentiment}


def extracting_hidden_entities_and_reviews_and_sentiment_prediction(file, vectorizer_filename, classifier_filename,
                                                                    data_json, list_of_entity_name, max_threshold=DEFAULT_THRESHOLD_CONST,min_threshold=DEFAULT_NEG_THRESHOLD_CONST):
    # print("Data:\n")
    # print(data_json, file, classifier_filename, vectorizer_filename, list_of_entity_name)
    # print("Type:\n")
    # print(type(data_json), type(file), type(classifier_filename), type(vectorizer_filename), type(list_of_entity_name))
    if (type(list_of_entity_name) != list):
        list_of_entity_name = ast.literal_eval(list_of_entity_name)
    df, ngram_vectorizer, svm = load_file_to_extract_hidden_entities(file, vectorizer_filename, classifier_filename,
                                                                     data_json)
    result = []
    # print(df.shape)
    # print("^^^^^^^^^^^^^^^^")

    for i, row in df.iterrows():
        neu_cor = pronoun_resolution(row)

        extracted_sentences, review_without_hidden_entity = extract_hidden_entities(neu_cor[0],
                                                                                    list_of_entity_name)  # (text[0])#
        review = row['Comment']

        # Predicting the whole review to get its class
        review_sentiment_prediction = classifying_the_sentence(review, ngram_vectorizer, svm)
        # review_predicted_class = review_sentiment_prediction[0]
        ##print('Negative :' ,review_predicted_class[0][0], 'Neutral :',review_predicted_class[0][1], 'Positive :',review_predicted_class[0][2])

        review_predicted_class_final = np.argmax(review_sentiment_prediction[0])

        if review_predicted_class_final == 0:
            review_predicted_class_final = 'Negative'
        elif review_predicted_class_final == 1:
            review_predicted_class_final = 'Neutral'
        elif review_predicted_class_final == 2:
            review_predicted_class_final = 'Positive'

        orginal_review = review
        orginal_review = orginal_review.replace(".", "")
        review_predicted_sentiment = get_sentiment_score_textblob(orginal_review, review_sentiment_prediction[0][0][2], max_threshold, min_threshold)

        dic_review = {"entity": row['Name'], "review": review,
                    #   "review_predicted_classes_probability_prelim": [('Negative :', review_predicted_class[0][0]),
                    #                                                   ('Neutral :', review_predicted_class[0][1]),
                    #                                                   ('Positive :', review_predicted_class[0][2])],
                      "review_predicted_sentiment_final": review_predicted_sentiment}
        ##print('The length of extracted_sentences', len(extracted_sentences))
        if len(extracted_sentences) == 0:
            msg = {"original_review": dic_review, "hidden_entity_sentence": None,
                   "review_without_extracted_sentences": dic_review}
            result.append(msg)

        # Checking if the review has more than 1 sentence if they do and contain hidden entity sentences they are predicted
        elif len(extracted_sentences) > 0:
            dic_hidden_entity_sentence = []
            for entity, sentence in extracted_sentences.items():
                ##print(entity,"","******","",sentence)
                sentence_sentiment_prediction = classifying_the_sentence(sentence, ngram_vectorizer, svm)
                sentence_predicted_class = sentence_sentiment_prediction[0]
                sentence_predicted_class_final = np.argmax(sentence_sentiment_prediction[0])

                if sentence_predicted_class_final == 0:
                    sentence_predicted_class_final = 'Negative'
                elif sentence_predicted_class_final == 1:
                    sentence_predicted_class_final = 'Neutral'
                elif sentence_predicted_class_final == 2:
                    sentence_predicted_class_final = 'Positive'

                #                 #print(max(pred_sentence[0][0]),class_prediction )

                review_without_sentence_sentiment_prediction = classifying_the_sentence(review_without_hidden_entity,
                                                                                        ngram_vectorizer, svm)
                review_without_sentence_predicted_class = review_without_sentence_sentiment_prediction[0]

                review_without_sentence_predicted_class_final = np.argmax(
                    review_without_sentence_sentiment_prediction[0])

                if review_without_sentence_predicted_class_final == 0:
                    review_without_sentence_predicted_class_final = 'Negative'
                elif review_without_sentence_predicted_class_final == 1:
                    review_without_sentence_predicted_class_final = 'Neutral'
                elif review_without_sentence_predicted_class_final == 2:
                    review_without_sentence_predicted_class_final = 'Positive'
                #                 #print(max(pred[0][0]),pred_class)
                # print(review_without_hidden_entity)
                without_hidden_entity = review_without_hidden_entity[0] if (len(review_without_hidden_entity) > 0) else ''
                without_hidden_entity = without_hidden_entity.replace(".", "")
                # print(without_hidden_entity)
                review_without_hidden_entity_sentence_predicted_sentiment = get_sentiment_score_textblob(
                    without_hidden_entity, review_without_sentence_sentiment_prediction[0][0][2], max_threshold, min_threshold)
                dic_rev_without_extracted_sentence = {"entity": row['Name'],
                                                      "review_without_hidden_entity_sentence_final": review_without_hidden_entity,
                                                      "review_without_sentence_predicted_class_probabilities_prelim": [
                                                          ('Negative :', review_without_sentence_predicted_class[0][0]),
                                                          ('Neutral :', review_without_sentence_predicted_class[0][1]),
                                                          (
                                                              'Positive :',
                                                              review_without_sentence_predicted_class[0][2])],
                                                      "review_without_hidden_entity_sentence_predicted_sentiment_final": review_without_hidden_entity_sentence_predicted_sentiment}
                hidden_entity_sentence_predicted_sentiment_result = get_sentiment_score_textblob(sentence, sentence_sentiment_prediction[0][0][2], max_threshold, min_threshold)
                # print(hidden_entity_sentence_predicted_sentiment_result)
                # print("-------------------------")

                dic_hidden_entity_sentence.append(
                    {"entity": row['Name'] + '_@_' + entity, "hidden_entity_sentence": sentence,
                     "hidden_entity_sentence_predicted_classes_probability_prelim": [
                         ('Negative :', sentence_predicted_class[0][0]), ('Neutral :', sentence_predicted_class[0][1]),
                         ('Positive :', sentence_predicted_class[0][2])],
                     "hidden_entity_sentence_predicted_sentiment_final": hidden_entity_sentence_predicted_sentiment_result})

            msg = {"original_review": dic_review, "hidden_entity_sentence": dic_hidden_entity_sentence,
                   "review_without_extracted_sentences": dic_rev_without_extracted_sentence}
            result.append(msg)

    return result


# Loading the file from which the common positive and negative features

def load_file_common_positive_negative_features(file, vectorizer_filename, classifier_filename, data_json):
    # If the given file is an excel sheet
    # print(vectorizer_filename)
    ngram_vectorizer = joblib.load(vectorizer_filename)
    svm = joblib.load(classifier_filename)

    if len(file) == 0 and len(data_json) == 0:
        print("Do provide a data file or json file")
        # Needs a check
        return "Do provide a data file or json file"
    if file and len(data_json) == 0:
        if (file.endswith('.xlx') or file.endswith('.xlsx')):
            # print(file)
            # print("*********************")
            df_common_entities = pd.read_excel(file)
        # If the given file is json
        elif file.endswith('.csv'):
            # print(file)
            df_common_entities = pd.read_csv(file)
        elif (file.endswith('.txt') or file.endswith('.json')):
            with open(file) as json_file:
                data = json.load(json_file)
                df_common_entities = pd.DataFrame.from_dict(data, orient='columns')
    if len(data_json) > 0:
        data_json = json.loads(data_json)
        # print(data_json)

        df_common_entities = json_normalize(data_json)
    #     df_common_entities = df_common_entities.drop(df_common_entities.index[df_common_entities['Rating']=='Rating'])
    #     df_common_entities.astype({'Rating':'float'}).dtypes
    if 'RepuGen Review' in df_common_entities.columns:
        df_common_entities = df_common_entities.rename(columns={"RepuGen Review": "Comment"})
    elif 'Review' in df_common_entities.columns:
        df_common_entities = df_common_entities.rename(columns={"Review": "Comment"})
    df_common_entities = df_common_entities.rename(columns={"Provider": "Name"})
    df_common_entities['Comment'] = df_common_entities['Comment'].astype(str)
    # df_common_entities= df_common_entities.iloc[:,4:]
    Rating_list = []
    for i, row in df_common_entities.iterrows():
        Rating = classifying_the_sentence(row['Comment'], ngram_vectorizer, svm)

        Rating_list.append(Rating[1][0])
        # print(Rating[1][0])
    df_common_entities['Rating'] = Rating_list

    return df_common_entities


def extract_hidden_entities_positive_negative_features(review, list_of_entity_name):
    extracted_sentences = {}
    review_without_hidden_entity = []
    review_wo = []
    sentence_count = 0
    for sentence in sent_tokenize(review):
        sentence_count += 1
        doc = nlp(sentence)
        for token in doc:
            # print(token.text, token.dep_)
            if token.dep_ == 'nsubj':
                token_index = token.i
                prev_token = doc[token_index - 1]
                try:
                    next_token = doc[token_index + 1]
                except Exception:
                    continue
                # CK sir logic
                # office staff current token = staff, previous token = office
                # office people current token = office, previous taken = xxx, next token = people
                if prev_token.text.lower() in list_of_entity_name:
                    if token.text.lower() in list_of_entity_name:
                        word = prev_token.text + " " + token.text
                        # print(word)
                        extracted_sentences[word] = sentence
                elif token.text.lower() in list_of_entity_name:
                    if next_token.text.lower() in list_of_entity_name:
                        word = token.text + " " + next_token.text
                        # print(word)
                        extracted_sentences[word] = sentence
                elif prev_token.text.lower() not in list_of_entity_name and next_token.text.lower() not in list_of_entity_name:
                    if token.text.lower() in list_of_entity_name:
                        # print(token.text)
                        extracted_sentences[token.text] = sentence

                # Extracting the entities only if they belong to this list of pre-defined words
                # KC: 11/6/20: The reason we extract 2 entities like office, staff if the input review has 'office staff' is
                # because of the prev_token logic -> validate this by commenting that prev_token and run.
                # Old pipeline:
                '''
                if token.text.lower() in list_of_entity_name:  # ['technicians','facility','assistant','desk','rep','lpn','office','rn','establishment','provider','specialists','employee','nurse','np','receptionist','pa','rn','staff','attendant','practitioner']:
                    extracted_sentences[token.text] = sentence
                if prev_token.text.lower() in list_of_entity_name:  # ['technicians','facility','assistant','desk','rep','lpn','office','rn','establishment','provider','specialists','employee','nurse','np','receptionist','pa','rn','staff','attendant','practitioner']:
                    extracted_sentences[prev_token.text] = sentence
                '''

    if sentence_count > 1:

        for i in extracted_sentences.values():
            #  print(i)
            # print("*^" * 10)
            review = review.replace(i, "")
        review_without_hidden_entity.append(review)

    return review_without_hidden_entity



# the lookup list extracts all the adjectives and adverbs from which the doctor names and other irrelevant features will be eliminated from.

def lookup_list(df):
    keyword_lookup = []
    for i, row in df.iterrows():
        for sentence in sent_tokenize(row['Comment']):
            doc = nlp(sentence)
            for word in doc:
                # Extracting only words if they are Adjectives or Adverbs
                if (word.pos_ == 'ADJ'):
                    if len(keyword_lookup) < 1:
                        keyword_lookup.append(str(word).lower())
                    elif len(keyword_lookup) >= 1:
                        if str(word).lower() not in keyword_lookup:
                            keyword_lookup.append(str(word).lower())
    return keyword_lookup


# Generating the common positive and negative attributes
def generate_common_positive_negative_attributes_test(file, top_x_attributes, vectorizer_filename,
                                                      classifier_filename, data_json):
    df = load_file_common_positive_negative_features(file, vectorizer_filename, classifier_filename, data_json)
    keyword_list = lookup_list(df)
    # print(keyword_list)
    df['Comment'] = df['Comment'].astype(str)
    df['Comment'] = data_preprocessing(df['Comment'])
    # df['Rating'] = classifying_lables(df)

    # Vectorises the data using unigram for optimal results
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(2, 2))
    ngram_vectorizer.fit(df['Comment'])
    X = ngram_vectorizer.transform(df['Comment'])
    # Checking whether there exist more than one review
    if (len(df.index) < 2):
        dic_common_positive_negative_features = {"Error": "Need a larger dataset"}

        return dic_common_positive_negative_features

    X_train, X_test, y_train, y_test = train_test_split(X, df['Rating'], train_size=0.8)

    # Checking whether there exist atleast one positive or negative comment in ytrain
    y_t = list(y_train)
    if 'Positive' not in y_t or 'Negative' not in y_t:
        dic_common_positive_negative_features = {"Error": "Need atleast one positive review and one Negative"}

        return dic_common_positive_negative_features

    maximum_acc = 0
    hyper_parameter = 0

    # Finding the best hyper parameter
    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        svm = LogisticRegression(C=c)

        svm.fit(X_train, y_train)

        if accuracy_score(y_test, svm.predict(X_test)) > maximum_acc:
            maximum_acc = accuracy_score(y_test, svm.predict(X_test))
            hyper_parameter = c
    # using logistic regression the features are predicted and extracted.
    final_count_ngram = LogisticRegression(C=hyper_parameter)
    final_count_ngram.fit(X, df['Rating'])
    feature_to_coef = {word: coef for word, coef in
                       zip(ngram_vectorizer.get_feature_names(), final_count_ngram.coef_[0])}
    # #print(feature_to_coef)

    positive_features = []
    for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True):
        if best_positive[1] > 0.0:
            positive_features.append(best_positive)

    count = 0
    positive_filtered_features = []
    for word in positive_features:
        words = word[0].split(' ')
        if (words[0] in keyword_list) | (words[1] in keyword_list):
            x = TextBlob(word[0])
            if (x.sentiment.polarity > 0.25):
                positive_filtered_features.append(word)
                count += 1
                if count == top_x_attributes:
                    break
    #         #print('number of positive_filtered_features is' , len(positive_filtered_features))

    negative_features = []
    for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1]):
        if best_negative[1] < 0.0:
            negative_features.append(best_negative)

    count = 0
    negative_filtered_features = []
    for word in negative_features:
        words = word[0].split(' ')
        if (words[0] in keyword_list) | (words[1] in keyword_list):

            x = TextBlob(word[0])
            if (x.sentiment.polarity < -0.25):
                negative_filtered_features.append(word)
                count += 1
                if count == top_x_attributes:
                    break

    dic_common_positive_negative_features = {"positive_filtered_features": positive_filtered_features,
                                             "negative_filtered_features": negative_filtered_features}

    return dic_common_positive_negative_features


def generate_common_positive_negative_attributes(file, top_x_attributes, Bigram_Boolean, vectorizer_filename,
                                                 classifier_filename, data_json):
    if Bigram_Boolean == "True":
        dic_common_positive_negative_features = generate_common_positive_negative_attributes_test(file,
                                                                                                  top_x_attributes,
                                                                                                  vectorizer_filename,
                                                                                                  classifier_filename,
                                                                                                  data_json)



    else:
        # print(Bigram_Boolean)
        df = load_file_common_positive_negative_features(file, vectorizer_filename, classifier_filename,
                                                         data_json)
        keyword_list = lookup_list(df)
        df['Comment'] = df['Comment'].astype(str)
        df['Comment'] = data_preprocessing(df['Comment'])

        # Extracting sentences based on the entities:

        # df['Rating'] = classifying_lables(df)

        # Vectorises the data using unigram for optimal results
        ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 1))
        ngram_vectorizer.fit(df['Comment'])
        X = ngram_vectorizer.transform(df['Comment'])
        # Checking whether we have more than one data
        if (len(df.index) < 2):
            dic_common_positive_negative_features = {"Error": "Need a larger dataset"}

            return dic_common_positive_negative_features
        X_train, X_test, y_train, y_test = train_test_split(
            X, df['Rating'], train_size=0.8)
        # print(y_train)
        # Checking whether both the positive and negative rating present in the y_train
        y_t = list(y_train)
        if 'Positive' not in y_t or 'Negative' not in y_t:
            dic_common_positive_negative_features = {"Error": "Need atleast one positive review and one Negative"}

            return dic_common_positive_negative_features
        maximum_acc = 0
        hyper_parameter = 0

        # Finding the best hyper parameter
        for c in [0.01, 0.05, 0.25, 0.5, 1]:
            svm = LogisticRegression(C=c)
            svm.fit(X_train, y_train)
            if accuracy_score(y_test, svm.predict(X_test)) > maximum_acc:
                maximum_acc = accuracy_score(y_test, svm.predict(X_test))
                hyper_parameter = c
        # using logistic regression the features are predicted and extracted.
        final_count_ngram = LogisticRegression(C=hyper_parameter)
        final_count_ngram.fit(X, df['Rating'])
        feature_to_coef = {word: coef for word, coef in
                           zip(ngram_vectorizer.get_feature_names(), final_count_ngram.coef_[0])}

        #         #print(feature_to_coef)
        #         break
        count = 0
        positive_features = []
        for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True):
            if best_positive[1] > 0.0:
                positive_features.append(best_positive)

        positive_filtered_features = []
        for word in positive_features:
            if word[0] in keyword_list:
                x = TextBlob(word[0])
                ##print(x.sentiment.polarity)
                if (x.sentiment.polarity > 0.25):
                    positive_filtered_features.append(word)
                    count += 1
                    if count == top_x_attributes:
                        break
                    #         #print('number of positive_filtered_features is' , len(positive_filtered_features))

        negative_features = []
        for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1]):
            if best_negative[1] < 0.0:
                negative_features.append(best_negative)

        count = 0
        negative_filtered_features = []
        for word in negative_features:
            if word[0] in keyword_list:
                x = TextBlob(word[0])
                if (x.sentiment.polarity < -0.25):
                    negative_filtered_features.append(word)
                    count += 1
                    if count == top_x_attributes:
                        break
        #         #print('number of negative_filtered_features is' , len(negative_filtered_features))

        dic_common_positive_negative_features = {"positive_filtered_features": positive_filtered_features,
                                                 "negative_filtered_features": negative_filtered_features}

    return dic_common_positive_negative_features


def get_dummy_data():
    # df= pd.read_excel("./upload/RepuGen Portal Reviews.xlsx")
    if ('RepuGen Portal Reviews.xlsx' in os.listdir("./upload")):
        print('File found')

    df = pd.read_excel("./upload/RepuGenAllData.xlsx")
    # print(df.head())
    if 'RepuGen Review' in df.columns:
        df = df.rename(columns={"RepuGen Review": "Comment"})
    elif 'Review' in df.columns:
        df = df.rename(columns={"Review": "Comment"})

    df.loc[df['Rating'] < 6, 'Rating'] = 0
    df.loc[df['Rating'] > 5, 'Rating'] = 1
    df['Rating'] = df['Rating'].astype(str)
    df['Rating'] = df['Rating'].replace(str(0), 'Negative')
    df['Rating'] = df['Rating'].replace(str(1), 'Positive')

    positive_dummy = df[df['Rating'] == 'Positive']
    positive_dummy = positive_dummy.reset_index(drop=True)

    negative_dummy = df[df['Rating'] == 'Negative']
    negative_dummy = negative_dummy.reset_index(drop=True)

    positive_dummy_df = positive_dummy.iloc[:1, 4:]
    negative_dummy_df = negative_dummy.iloc[:1, 4:]

    # vectorizer_filename = joblib.load(vectorizer_filename)
    # classifier_filename = joblib.load(classifier_filename)

    return positive_dummy_df, negative_dummy_df


def load_file_entity_specific_features(file, vectorizer_filename, classifier_filename, data_json, list_of_entity_name):
    pos_count = 0
    neg_count = 0
    ngram_vectorizer = joblib.load(vectorizer_filename)
    svm = joblib.load(classifier_filename)

    # Converting the json file to a dataframe
    # print("--------------------------------------")

    if len(file) == 0 and len(data_json) == 0:
        print("Do provide a data file or json file")
        # Needs a check
        return "Do provide a data file or json file"
    if file and len(data_json) == 0:

        if (file.endswith('.xlx') or file.endswith('.xlsx')):
            # print(file)
            df_entity_specific = pd.read_excel(file)
            # print(df_entity_specific.head())
        elif file.endswith('.csv'):
            df_entity_specific = pd.read_csv(file)

        elif (file.endswith('.txt') or file.endswith('.json')):
            with open(file) as json_file:
                data = json.load(json_file)
                df_entity_specific = pd.DataFrame.from_dict(data, orient='columns')
    if len(data_json) > 0:
        data_json = json.loads(data_json)
        # print(data_json)
        df_entity_specific = json_normalize(data_json)

        # df_entity_specific = pd.DataFrame.from_dict(file, orient='columns')
    #     df_entity_specific = df_entity_specific.drop(df_entity_specific.index[df_entity_specific['Rating']=='Rating'])
    #     df_entity_specific.astype({'Rating':'float'}).dtypes
    if 'RepuGen Review' in df_entity_specific.columns:
        df_entity_specific = df_entity_specific.rename(columns={"RepuGen Review": "Comment"})
    elif 'Review' in df_entity_specific.columns:
        df_entity_specific = df_entity_specific.rename(columns={"Review": "Comment"})
    #     df_entity_specific = df_entity_specific.rename(columns={"Provider":"Name"})
    df_entity_specific['Comment'] = df_entity_specific['Comment'].astype(str)
    # df_entity_specific= df_entity_specific.

    df_positive_dummy, df_negative_dummy = get_dummy_data()
    df_positive_dummy = df_positive_dummy.append([df_positive_dummy] * 80, ignore_index=True)
    df_negative_dummy = df_negative_dummy.append([df_negative_dummy] * 20, ignore_index=True)

    positive_keyword_list = lookup_list(df_positive_dummy)
    negative_keyword_list = lookup_list(df_negative_dummy)
    regular_keyword_list = lookup_list(df_entity_specific)

    Rating_list = []
    final_sentence = []
    for i, row in df_entity_specific.iterrows():

        review_without_sentence = [
            extract_hidden_entities_positive_negative_features(row['Comment'], list_of_entity_name)]

        if len(review_without_sentence[0]) > 0:
            final_sentence.append(review_without_sentence[0][0])
        else:
            final_sentence.append(row['Comment'])

        Rating = classifying_the_sentence(row['Comment'], ngram_vectorizer, svm)
        Rating_list.append(Rating[1][0])
        # #print(Rating_list)
    df_entity_specific['Rating'] = Rating_list

    Rating_list = []
    final_sentence = []
    for i, row in df_positive_dummy.iterrows():
        review_without_sentence = [
            extract_hidden_entities_positive_negative_features(row['Comment'], list_of_entity_name)]

        if len(review_without_sentence[0]) > 0:
            final_sentence.append(review_without_sentence[0][0])
        else:
            final_sentence.append(row['Comment'])

        Rating = classifying_the_sentence(row['Comment'], ngram_vectorizer, svm)

        Rating_list.append(Rating[1][0])
    df_positive_dummy['Rating'] = Rating_list

    Rating_list = []
    final_sentence = []
    for i, row in df_negative_dummy.iterrows():
        review_without_sentence = [
            extract_hidden_entities_positive_negative_features(row['Comment'], list_of_entity_name)]

        if len(review_without_sentence[0]) > 0:
            final_sentence.append(review_without_sentence[0][0])
        else:
            final_sentence.append(row['Comment'])

        Rating = classifying_the_sentence(row['Comment'], ngram_vectorizer, svm)

        Rating_list.append(Rating[1][0])
    df_negative_dummy['Rating'] = Rating_list

    for i, row in df_entity_specific.iterrows():
        if row['Rating'] == 'Positive':
            pos_count += 1
    df_positive_limiter = pos_count

    for i, row in df_entity_specific.iterrows():
        if row['Rating'] == 'Negative':
            neg_count += 1
    df_negative_limiter = neg_count

    #     #print(df_entity_specific.head())

    # if len(df_entity_specific)

    df_entity_specific_final = pd.concat([df_entity_specific, df_positive_dummy.iloc[df_positive_limiter:, :],
                                          df_negative_dummy.iloc[df_negative_limiter:, :]], axis=0)
    df_entity_specific_final = df_entity_specific_final.reset_index(drop=True)
    df_entity_specific_final = pd.DataFrame(df_entity_specific_final.iloc[:, :])

    # #print(df_entity_specific_final.head())

    return df_entity_specific_final, positive_keyword_list, negative_keyword_list, regular_keyword_list


def generate_entity_specific_positive_negative_attributes_test(file, top_x_attributes, vectorizer_filename,
                                                               classifier_filename, data_json, list_of_entity_name):
    # print("Bigram ---------------------------------------------")
    df, positive_keyword_list, negative_keyword_list, regular_keyword_list = load_file_entity_specific_features(file,
                                                                                                                vectorizer_filename,
                                                                                                                classifier_filename,
                                                                                                                data_json,
                                                                                                                list_of_entity_name)
    #     keyword_list = lookup_list(df)
    df['Comment'] = df['Comment'].astype(str)
    df['Comment'] = data_preprocessing(df['Comment'])
    # print(df['Comment'])
    #     df['Rating'] = classfying_entity_specific_lables(df)
    ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(2, 2))
    ngram_vectorizer.fit(df['Comment'])

    # The data is getting vectorised
    X = ngram_vectorizer.transform(df['Comment'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, df['Rating'], train_size=0.8)
    maximum_acc = 0
    hyper_parameter = 0

    # Finding the best hyper parameter
    for c in [0.01, 0.05, 0.25, 0.5, 1]:
        svm = LogisticRegression(C=c)
        svm.fit(X_train, y_train)
        if accuracy_score(y_test, svm.predict(X_test)) > maximum_acc:
            maximum_acc = accuracy_score(y_test, svm.predict(X_test))
            hyper_parameter = c
    # using logistic regression the features are predicted and extracted.
    final_count_ngram = LogisticRegression(C=hyper_parameter)
    final_count_ngram.fit(X, df['Rating'])
    feature_to_coef = {word: coef for word, coef in
                       zip(ngram_vectorizer.get_feature_names(), final_count_ngram.coef_[0])}
    # print(feature_to_coef)
    # print("*-*"*10)
    count = 0
    positive_features = []
    for best_positive in sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True):
        if best_positive[1] > 0.0:
            positive_features.append(best_positive)

            # print(positive_features)
    positive_filtered_features = []
    # print(regular_keyword_list)
    for word in positive_features:
        # print("word: ", word)
        # print("positive_keyword_list:", positive_keyword_list)
        # print("regular keyword:", regular_keyword_list)
        words = word[0].split(' ')
        if (words[0] in positive_keyword_list) | (words[1] in positive_keyword_list) | (
                words[0] in regular_keyword_list) | (words[1] in regular_keyword_list):
            # print(word[0])
            x = TextBlob(word[0])
            ##print(x.sentiment.polarity)
            if (x.sentiment.polarity > 0.25):
                positive_filtered_features.append(word)
                count += 1
                if count == top_x_attributes:
                    break
                #         #print('number of positive_filtered_features is' , len(positive_filtered_features))

    negative_features = []
    for best_negative in sorted(feature_to_coef.items(), key=lambda x: x[1]):
        if best_negative[1] < 0.0:
            negative_features.append(best_negative)

    count = 0
    negative_filtered_features = []
    for word in negative_features:
        words = word[0].split(' ')
        if (words[0] in positive_keyword_list) | (words[1] in positive_keyword_list) | (
                words[0] in regular_keyword_list) | (words[1] in regular_keyword_list):
            x = TextBlob(word[0])
            if (x.sentiment.polarity < -0.25):
                negative_filtered_features.append(word)
                count += 1
                if count == top_x_attributes:
                    break

    dic_entity_specific_positive_negative_features = {"positive_filtered_features": positive_filtered_features,
                                                      "negative_filtered_features": negative_filtered_features}

    return dic_entity_specific_positive_negative_features


# Entity Specific positive and negative features are extracted here.
# Entity Specific positive and negative features are extracted here.
def generate_entity_specific_positive_negative_attributes(file, top_x_attributes, Bigram_boolean,
                                                          vectorizer_filename, classifier_filename, data_json,
                                                          list_of_entity_name):
    if (type(list_of_entity_name) != list):
        list_of_entity_name = ast.literal_eval(list_of_entity_name)
    if Bigram_boolean == 'True':
        dic_entity_specific_positive_negative_features = generate_entity_specific_positive_negative_attributes_test(
            file, top_x_attributes, vectorizer_filename, classifier_filename, data_json, list_of_entity_name)

    else:
        df, positive_keyword_list, negative_keyword_list, regular_keyword_list = load_file_entity_specific_features(
            file, vectorizer_filename, classifier_filename, data_json, list_of_entity_name)
        #         keyword_list = lookup_list(df)
        df['Comment'] = df['Comment'].astype(str)
        df['Comment'] = data_preprocessing(df['Comment'])
        # #print(df['Rating'].unique)
        #         df['Rating'] = classfying_entity_specific_lables(df)
        ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 1))
        ngram_vectorizer.fit(df['Comment'])

        # The data is getting vectorised
        X = ngram_vectorizer.transform(df['Comment'])
        X_train, X_test, y_train, y_test = train_test_split(
            X, df['Rating'], train_size=0.8)
        maximum_acc = 0
        hyper_parameter = 0

        # finding the best hyper paramater

        for c in [0.01, 0.05, 0.25, 0.5, 1]:
            svm = LogisticRegression(C=c)
            svm.fit(X_train, y_train)

            if accuracy_score(y_test, svm.predict(X_test)) > maximum_acc:
                maximum_acc = accuracy_score(y_test, svm.predict(X_test))
                hyper_parameter = c

        # Using Logistic Regression the best featrues are predicted

        final_count_ngram = LogisticRegression(C=hyper_parameter)
        final_count_ngram.fit(X, df['Rating'])
        feature_to_coef = {word: coef for word, coef in
                           zip(ngram_vectorizer.get_feature_names(), final_count_ngram.coef_[0])}
        # print(feature_to_coef)

        positive_filtered_features = []

        negative_filtered_features = []
        for text in feature_to_coef:
            # print(text)
            result = get_text_blob_score(text)
            # {"score":score,"sentiment":sentiment}
            if (result["sentiment"] == "positive"):
                positive_filtered_features.append(result)
            elif (result["sentiment"] == "negative"):
                negative_filtered_features.append(result)

        dic_entity_specific_positive_negative_features = {"positive_features": positive_filtered_features,
                                                          "negative_features": negative_filtered_features}

    return dic_entity_specific_positive_negative_features


def Extract_Adjectives(review):
    keyword_lookup = []
    flag = 0
    for sentence in sent_tokenize(review):
        doc = nlp(sentence)
        for word in doc:
            # print(word, word.pos_)

            if (word.pos_ == 'ADJ'):
                token_index = word.i
                if token_index >= 1:
                    fir_token = doc[token_index - 1]
                else:
                    fir_token = doc[token_index]
                if token_index >= 2:
                    sec_token = doc[token_index - 2]
                else:
                    sec_token = doc[token_index]

                if sec_token.dep_ == 'neg':
                    if sec_token.text == "nt" or sec_token.text == "n’t" or sec_token.text == "n't":
                        word = "not" + " " + word.text
                    else:
                        word = sec_token.text + " " + word.text

                elif (fir_token.dep_ == 'neg'):
                    if fir_token.text == "nt" or fir_token.text == "n’t" or fir_token.text == "n't":
                        word = "not" + " " + word.text
                    else:
                        word = fir_token.text + " " + word.text

                if len(keyword_lookup) < 1:
                    keyword_lookup.append(str(word).lower())
                elif len(keyword_lookup) >= 1:
                    if str(word).lower() not in keyword_lookup:
                        keyword_lookup.append(str(word).lower())
    return keyword_lookup


# Judges whether adjective is positive or negative and then puts in a list
def Extract_Attributes(review, atr_thres_pos, atr_thres_neg, global_pos_list, global_neg_list):
    pos_adj = []
    neg_adj = []
    adj_list = Extract_Adjectives(review)
    for adj in adj_list:
        zen = TextBlob(adj)
        for sentence in zen.sentences:
            if sentence.sentiment.polarity >= atr_thres_pos:
                if adj in global_pos_list:
                    score = sentence.sentiment.polarity
                    score = round_fload_val(score)
                    #score = round(score,2)
                    adj = adj + "|" + str(score)
                    pos_adj.append(adj)
                else:
                    res = [i for i in global_pos_list if adj in i and "|" in i]
                    for i in range(len(res)):
                        bs = res[i]
                        pos_res = res[i].split("|")
                        if pos_res[0] == adj:
                            adj = bs
                            pos_adj.append(adj)
            if sentence.sentiment.polarity < atr_thres_neg:
                if adj in global_neg_list:
                    score = sentence.sentiment.polarity
                    score = round_fload_val(score)
                   # score = round(score, 2)
                    adj = adj + "|" + str(score)
                    neg_adj.append(adj)
                else:
                    res = [i for i in global_neg_list if adj in i and "|" in i]
                    for i in range(len(res)):
                        bs = res[i]
                        neg_res = res[i].split("|")
                        if neg_res[0] == adj:
                            adj = bs
                            neg_adj.append(adj)
    return pos_adj, neg_adj


# Converts the file to df
def jsontodf(data_json, file):
    if len(file) == 0 and len(data_json) == 0:
        print("Do provide a data file or json file")
        # Needs a check
        return "Do provide a data file or json file"
    if file and len(data_json) == 0:

        if (file.endswith('.xlx') or file.endswith('.xlsx')):
            # print(file)
            df_entity_specific = pd.read_excel(file)
            # print(df_entity_specific.head())
        elif file.endswith('.csv'):
            df_entity_specific = pd.read_csv(file)

        elif (file.endswith('.txt') or file.endswith('.json')):
            with open(file) as json_file:
                data = json.load(json_file)
                df_entity_specific = pd.DataFrame.from_dict(data, orient='columns')
    if len(data_json) > 0:
        data_json = json.loads(data_json)
      #  print(data_json)
        df_entity_specific = json_normalize(data_json)
   # print(df_entity_specific)
    if 'RepuGen Review' in df_entity_specific.columns:
        df_entity_specific = df_entity_specific.rename(columns={"RepuGen Review": "Comment"})
    elif 'Review' in df_entity_specific.columns:
        df_entity_specific = df_entity_specific.rename(columns={"Review": "Comment"})
    df_entity_specific = df_entity_specific.rename(columns={"Provider": "Name"})
    df_entity_specific['Comment'] = df_entity_specific['Comment'].astype(str)

    return df_entity_specific

#Newly modified codes -> starts

def remove_specialchars(text):
    text = text.lower()
    text = text.replace("dr.", "dr ")
    text = text.replace("dr .", "dr ")
    #text = text.replace("dr", "")
    text = text.replace("Dr.", "Dr ")
    text = text.replace("Dr .", "Dr ")
    #text = text.replace("Dr", "")
    text = text.replace(",", "")
    text = text.replace("-", "")
    text = text.replace("&","and")
    text = text.replace("ain't", "am not")
    text = text.replace("aren't", "are not")
    text = text.replace("can't", "cannot")
    text = text.replace("can't've", "cannot have")
    text = text.replace("'cause", "because")
    text = text.replace("could've", "could have")
    text = text.replace("couldn't", "could not")
    text = text.replace("couldn't've", "could not have")
    text = text.replace("didn't", "did not")
    text = text.replace("doesn't", "does not")
    text = text.replace("don't", "do not")
    text = text.replace("hadn't", "had not")
    text = text.replace("hadn't've", "had not have")
    text = text.replace("hasn't", "has not")
    text = text.replace("haven't", "have not")
    text = text.replace("he'd", "he had")
    text = text.replace("he'd've", "he would have")
    text = text.replace("he'll", "he will")
    text = text.replace("he'll've", "he will have")
    text = text.replace("he's", "he has")
    text = text.replace("how'd", "how did")
    text = text.replace("how'd'y", "how do you")
    text = text.replace("how'll", "how will")
    text = text.replace("how's", "how has")
    text = text.replace("i'd", "I had")
    text = text.replace("i'd've", "I would have")
    text = text.replace("i'll", "I shall")
    text = text.replace("i'll've", "I shall have")
    text = text.replace("i'm", "I am")
    text = text.replace("i've", "I have")
    text = text.replace("isn't", "is not")
    text = text.replace("it'd", "it had")
    text = text.replace("it'd've", "it would have")
    text = text.replace("it'll", "it shall")
    text = text.replace("it'll've", "it shall have")
    text = text.replace("it's", "it has")
    text = text.replace("let's", "let us")
    text = text.replace("ma'am", "madam")
    text = text.replace("mayn't", "may not")
    text = text.replace("might've", "might have")
    text = text.replace("mightn't", "might not")
    text = text.replace("mightn't've", "might not have")
    text = text.replace("must've", "must have")
    text = text.replace("mustn't", "must not")
    text = text.replace("mustn't've", "must not have")
    text = text.replace("needn't", "need not")
    text = text.replace("needn't've", "need not have")
    text = text.replace("o'clock", "of the clock")
    text = text.replace("oughtn't", "ought not")
    text = text.replace("oughtn't've", "ought not have")
    text = text.replace("shan't", "shall not")
    text = text.replace("sha'n't", "shall not")
    text = text.replace("shan't've", "shall not have")
    text = text.replace("she'd", "she had")
    text = text.replace("she'd've", "she would have")
    text = text.replace("she'll", "she shall")
    text = text.replace("she'll've", "she shall have")
    text = text.replace("she's", "she has")
    text = text.replace("should've", "should have")
    text = text.replace("shouldn't", "should not")
    text = text.replace("shouldn't've", "should not have")
    text = text.replace("so've", "so have")
    text = text.replace("so's", "so as")
    text = text.replace("that'd", "that would")
    text = text.replace("that'd've", "that would have")
    text = text.replace("that's", "that has")
    text = text.replace("there'd", "there had")
    text = text.replace("there'd've", "there would have")
    text = text.replace("there's", "there has")
    text = text.replace("they'd", "they had")
    text = text.replace("they'd've", "they would have")
    text = text.replace("they'll", "they shall")
    text = text.replace("they'll've", "they shall have")
    text = text.replace("they're", "they are")
    text = text.replace("they've", "they have")
    text = text.replace("to've", "to have")
    text = text.replace("wasn't", "was not")
    text = text.replace("we'd", "we had")
    text = text.replace("we'd've", "we would have")
    text = text.replace("we'll", "we will")
    text = text.replace("we'll've", "we will have")
    text = text.replace("we're", "we are")
    text = text.replace("we've", "we have")
    text = text.replace("weren't", "were not")
    text = text.replace("what'll", "what shall")
    text = text.replace("what'll've", "what shall have")
    text = text.replace("what're", "what are")
    text = text.replace("what's", "what has")
    text = text.replace("what've", "what have")
    text = text.replace("when's", "when has")
    text = text.replace("when've", "when have")
    text = text.replace("where'd", "where did")
    text = text.replace("where's", "where has")
    text = text.replace("where've", "where have")
    text = text.replace("who'll", "who shall")
    text = text.replace("who'll've", "who shall have")
    text = text.replace("who's", "who has")
    text = text.replace("who've", "who have")
    text = text.replace("why's", "why has")
    text = text.replace("why've", "why have")
    text = text.replace("will've", "will have")
    text = text.replace("won't", "will not")
    text = text.replace("won't've", "will not have")
    text = text.replace("would've", "would have")
    text = text.replace("wouldn't", "would not")
    text = text.replace("wouldn't've", "would not have")
    text = text.replace("y'all", "you all")
    text = text.replace("y'all'd", "you all would")
    text = text.replace("y'all'd've", "you all would have")
    text = text.replace("y'all're", "you all are")
    text = text.replace("y'all've", "you all have")
    text = text.replace("you'd", "you had")
    text = text.replace("you'd've", "you would have")
    text = text.replace("you'll", "you shall")
    text = text.replace("you'll've", "you shall have")
    text = text.replace("you're", "you are")
    text = text.replace("you've", "you have")
    text = text.replace("ain’t", "am not")
    text = text.replace("aren’t", "are not")
    text = text.replace("can’t", "cannot")
    text = text.replace("can’t’ve", "cannot have")
    text = text.replace("’cause", "because")
    text = text.replace("could’ve", "could have")
    text = text.replace("couldn’t", "could not")
    text = text.replace("couldn’t’ve", "could not have")
    text = text.replace("didn’t", "did not")
    text = text.replace("doesn’t", "does not")
    text = text.replace("don’t", "do not")
    text = text.replace("hadn’t", "had not")
    text = text.replace("hadn’t've", "had not have")
    text = text.replace("hasn’t", "has not")
    text = text.replace("haven’t", "have not")
    text = text.replace("he’d", "he had")
    text = text.replace("he’d’ve", "he would have")
    text = text.replace("he’ll", "he will")
    text = text.replace("he’ll’ve", "he will have")
    text = text.replace("he’s", "he has")
    text = text.replace("how’d", "how did")
    text = text.replace("how’d’y", "how do you")
    text = text.replace("how’ll", "how will")
    text = text.replace("how’s", "how has")
    text = text.replace("i’d", "I had")
    text = text.replace("i’d’ve", "I would have")
    text = text.replace("i’ll", "I shall")
    text = text.replace("i’ll’ve", "I shall have")
    text = text.replace("i’m", "I am")
    text = text.replace("i’ve", "I have")
    text = text.replace("isn’t", "is not")
    text = text.replace("it’d", "it had")
    text = text.replace("it’d’ve", "it would have")
    text = text.replace("it’ll", "it shall")
    text = text.replace("it’ll’ve", "it shall have")
    text = text.replace("it’s", "it has")
    text = text.replace("let’s", "let us")
    text = text.replace("ma’am", "madam")
    text = text.replace("mayn’t", "may not")
    text = text.replace("might’ve", "might have")
    text = text.replace("mightn’t", "might not")
    text = text.replace("mightn’t’ve", "might not have")
    text = text.replace("must’ve", "must have")
    text = text.replace("mustn’t", "must not")
    text = text.replace("mustn’t’ve", "must not have")
    text = text.replace("needn’t", "need not")
    text = text.replace("needn’t’ve", "need not have")
    text = text.replace("o’clock", "of the clock")
    text = text.replace("oughtn’t", "ought not")
    text = text.replace("oughtn’t’ve", "ought not have")
    text = text.replace("shan’t", "shall not")
    text = text.replace("sha’n’t", "shall not")
    text = text.replace("shan’t’ve", "shall not have")
    text = text.replace("she’d", "she had")
    text = text.replace("she’d’ve", "she would have")
    text = text.replace("she’ll", "she shall")
    text = text.replace("she’ll’ve", "she shall have")
    text = text.replace("she’s", "she has")
    text = text.replace("should’ve", "should have")
    text = text.replace("shouldn’t", "should not")
    text = text.replace("shouldn’t’ve", "should not have")
    text = text.replace("so’ve", "so have")
    text = text.replace("so’s", "so as")
    text = text.replace("that’d", "that would")
    text = text.replace("that’d’ve", "that would have")
    text = text.replace("that’s", "that has")
    text = text.replace("there’d", "there had")
    text = text.replace("there’d’ve", "there would have")
    text = text.replace("there’s", "there has")
    text = text.replace("they’d", "they had")
    text = text.replace("they’d’ve", "they would have")
    text = text.replace("they’ll", "they shall")
    text = text.replace("they’ll’ve", "they shall have")
    text = text.replace("they’re", "they are")
    text = text.replace("they’ve", "they have")
    text = text.replace("to’ve", "to have")
    text = text.replace("wasn’t", "was not")
    text = text.replace("we’d", "we had")
    text = text.replace("we’d’ve", "we would have")
    text = text.replace("we’ll", "we will")
    text = text.replace("we’ll’ve", "we will have")
    text = text.replace("we’re", "we are")
    text = text.replace("we’ve", "we have")
    text = text.replace("weren’t", "were not")
    text = text.replace("what’ll", "what shall")
    text = text.replace("what’ll’ve", "what shall have")
    text = text.replace("what’re", "what are")
    text = text.replace("what’s", "what has")
    text = text.replace("what’ve", "what have")
    text = text.replace("when’s", "when has")
    text = text.replace("when've", "when have")
    text = text.replace("where’d", "where did")
    text = text.replace("where’s", "where has")
    text = text.replace("where’ve", "where have")
    text = text.replace("who’ll", "who shall")
    text = text.replace("who’ll’ve", "who shall have")
    text = text.replace("who’s", "who has")
    text = text.replace("who’ve", "who have")
    text = text.replace("why’s", "why has")
    text = text.replace("why’ve", "why have")
    text = text.replace("will’ve", "will have")
    text = text.replace("won’t", "will not")
    text = text.replace("won’t’ve", "will not have")
    text = text.replace("would’ve", "would have")
    text = text.replace("wouldn’t", "would not")
    text = text.replace("wouldn’t’ve", "would not have")
    text = text.replace("y’all", "you all")
    text = text.replace("y’all’d", "you all would")
    text = text.replace("y’all’d’ve", "you all would have")
    text = text.replace("y’all’re", "you all are")
    text = text.replace("y’all’ve", "you all have")
    text = text.replace("you’d", "you had")
    text = text.replace("you’d’ve", "you would have")
    text = text.replace("you’ll", "you shall")
    text = text.replace("you’ll’ve", "you shall have")
    text = text.replace("you’re", "you are")
    text = text.replace("you’ve", "you have")
    return text  # .correc,t()

#Danger: Put error handling soon!!

def split_sentence_simplified(blob):
    split_sentences = ""
    list_split = []
    for sentence in blob.sentences:
        try:
            word_list = str(sentence).split(" ")
            # Removing the null values from the list
            word_list = [string for string in word_list if string != ""]
            word = ' '.join(word_list)
            doc = nlp(word)
            pos_list = [token.pos_ for token in doc]
            conj_indexes = [n for n, x in enumerate(pos_list) if x == 'CCONJ']
            #word_list = str(sentence).split(" ")
            #Removing the null values from the list
            #word_list = [string for string in word_list if string != ""]
            for i in conj_indexes:
                # Getting an index out of bounds error sometimes
                if(len(word_list) <= i):
                   continue
                # Dr. Biorkman was very warm and attentive -> should not split here
                if(len(pos_list[i + 1:])>2):
                    #print("Detected the conjunction and the index is "+str(i))
                    #and then prescribed both an antibiotic and a steroid together -> shouldn't split on both
                    if(word_list[i] != 'both'):
                        #print("The word is not both")
                       #print(word_list[i].lower())
                        #You get in to your appointment right away they are quick and efficient.  Zinar was kind very quick didnt really spend to much time with me got in and got out. The office staff is rude and the office was dirty.
                        if(word_list[i].lower() in conj_list):
                           # print("word is in the list "+word_list[i])
                            word_list[i] = "."
            split_sentences = " ".join(word_list)
            list_split.append(split_sentences)
        except Exception as e:
            print(e)
    split_sentences = " ".join(list_split)
    #print(split_sentences)
    return split_sentences
#newly modified code -> ends
# Extractes the hidden entity and score

def Extract_Score(df, classifier_filename, vectorizer_filename, list_of_entity_name, max_pos_threshold, max_neg_threshold):
    result = []
    ngram_vectorizer = joblib.load(vectorizer_filename)
    svm = joblib.load(classifier_filename)
    for i, row in df.iterrows():
        processed_text = nlp_core_dataProces(row['Name'], row['Comment'])
        # print("processed_text-",processed_text)
        extracted_sentences, review_without_hidden_entity = extract_hidden_entities(processed_text, list_of_entity_name)  

        # print("Extracted_Sent-",extracted_sentences)
        # print("----------------")
        # print("review_without_hidden_entity-",review_without_hidden_entity)
        # print("----------------")
        #review = row['Comment']
        review = processed_text
        col = list(df.columns)
        root = {}
        for i in col:
            root[i] = row[i]
        # Predicting the whole review to get its class
        review_sentiment_prediction = classifying_the_sentence(review, ngram_vectorizer, svm)
        print("review_sentiment_prediction-",review_sentiment_prediction[0][0][2])
        review_predicted_class = review_sentiment_prediction[0]

        review_predicted_class_final = np.argmax(review_sentiment_prediction[0])

        if review_predicted_class_final == 0:
            review_predicted_class_final = 'Negative'
        elif review_predicted_class_final == 1:
            review_predicted_class_final = 'Neutral'
        elif review_predicted_class_final == 2:
            review_predicted_class_final = 'Positive'

        orginal_review = review
        orginal_review = orginal_review.replace(".", "")
        review_predicted_sentiment = get_sentiment_score_textblob(orginal_review, review_sentiment_prediction[0][0][2], max_pos_threshold, max_neg_threshold)

        dic_review = {"entity": row['Name'], "review": review, "Positive_attributes": None, "Negative_attributes": None,
                      "review_predicted_classes_probability_prelim": [('Negative :', review_predicted_class[0][0]),
                                                                      ('Neutral :', review_predicted_class[0][1]),
                                                                      ('Positive :', review_predicted_class[0][2])],
                      "review_predicted_sentiment_final": review_predicted_sentiment}
        if len(extracted_sentences) == 0:
            msg = {"aroot": root, "original_review": dic_review, "hidden_entity_sentence": None,
                   "review_without_extracted_sentences": dic_review}
            result.append(msg)

        # Checking if the review has more than 1 sentence if they do and contain hidden entity sentences they are predicted
        elif len(extracted_sentences) > 0:
            dic_hidden_entity_sentence = []
            for entity, sentence in extracted_sentences.items():
                sentence_sentiment_prediction = classifying_the_sentence(sentence, ngram_vectorizer, svm)
                sentence_predicted_class = sentence_sentiment_prediction[0]
                sentence_predicted_class_final = np.argmax(sentence_sentiment_prediction[0])

                if sentence_predicted_class_final == 0:
                    sentence_predicted_class_final = 'Negative'
                elif sentence_predicted_class_final == 1:
                    sentence_predicted_class_final = 'Neutral'
                elif sentence_predicted_class_final == 2:
                    sentence_predicted_class_final = 'Positive'

                review_without_sentence_sentiment_prediction = classifying_the_sentence(review_without_hidden_entity,
                                                                                        ngram_vectorizer, svm)
                review_without_sentence_predicted_class = review_without_sentence_sentiment_prediction[0]

                review_without_sentence_predicted_class_final = np.argmax(
                    review_without_sentence_sentiment_prediction[0])

                if review_without_sentence_predicted_class_final == 0:
                    review_without_sentence_predicted_class_final = 'Negative'
                elif review_without_sentence_predicted_class_final == 1:
                    review_without_sentence_predicted_class_final = 'Neutral'
                elif review_without_sentence_predicted_class_final == 2:
                    review_without_sentence_predicted_class_final = 'Positive'
                without_hidden_entity = review_without_hidden_entity
                without_hidden_entity = without_hidden_entity.replace(".", "")
                review_without_hidden_entity_sentence_predicted_sentiment = get_sentiment_score_textblob(
                    without_hidden_entity, review_without_sentence_sentiment_prediction[0][0][2], max_pos_threshold, max_neg_threshold)
                dic_rev_without_extracted_sentence = {"entity": row['Name'],
                                                      "review_without_hidden_entity_sentence_final": review_without_hidden_entity,
                                                      "Positive_attributes": None, "Negative_attributes": None,
                                                      "review_without_sentence_predicted_class_probabilities_prelim": [
                                                          ('Negative :', review_without_sentence_predicted_class[0][0]),
                                                          ('Neutral :', review_without_sentence_predicted_class[0][1]),
                                                          (
                                                              'Positive :',
                                                              review_without_sentence_predicted_class[0][2])],
                                                      "review_without_hidden_entity_sentence_predicted_sentiment_final": review_without_hidden_entity_sentence_predicted_sentiment}
                hidden_entity_sentence_predicted_sentiment_result = get_sentiment_score_textblob(sentence, sentence_sentiment_prediction[0][0][2], max_pos_threshold, max_neg_threshold)
                dic_hidden_entity_sentence.append(
                    {"entity": row['Name'] + '_@_' + entity, "hidden_entity_sentence": sentence,
                     "Positive_attributes": None, "Negative_attributes": None,
                     "hidden_entity_sentence_predicted_classes_probability_prelim": [
                         ('Negative :', sentence_predicted_class[0][0]), ('Neutral :', sentence_predicted_class[0][1]),
                         ('Positive :', sentence_predicted_class[0][2])],
                     "hidden_entity_sentence_predicted_sentiment_final": hidden_entity_sentence_predicted_sentiment_result})

            msg = {"aroot": root, "original_review": dic_review, "hidden_entity_sentence": dic_hidden_entity_sentence,
                   "review_without_extracted_sentences": dic_rev_without_extracted_sentence}
            result.append(msg)

    return result

# Extracts hidden entity, score and also displays positive and negative attributes
def ExtractHidden_Score_Generate_Attributes(data_json, data_file, classifier, vectorizer, atr_thres_pos, atr_thres_neg,
                                            global_pos_list, global_neg_list, list_of_entity_name, max_pos_threshold, max_neg_threshold):
    if type(global_neg_list) != list:
        global_neg_list = ast.literal_eval(global_neg_list)
        global_pos_list = ast.literal_eval(global_pos_list)
    if type(list_of_entity_name) != list:
        list_of_entity_name = ast.literal_eval(list_of_entity_name)

    dfinput = jsontodf(data_json, data_file)
    #  --dfinput[provider, comments]

    json_input_extracted_scored = Extract_Score(dfinput, classifier, vectorizer, list_of_entity_name, max_pos_threshold, max_neg_threshold)
    length = len(dfinput.index)
    var = json.dumps(json_input_extracted_scored, default=str)
    try:
        data = json.loads(var)
    # print ("Is valid json? true")
    except ValueError as e:
        print("Is valid json? false")

    orgrev = []
    hes = []
    heslen = []
    rwes = []
    for i in range(length):
        pos_list = []
        neg_list = []
        orgrev.append(data[i]['original_review']['review'])
        p, n = Extract_Attributes(orgrev[i], atr_thres_pos, atr_thres_neg, global_pos_list, global_neg_list)
        temp = data[i]['original_review']
        temp['Positive_attributes'] = ','.join(p)
        temp['Negative_attributes'] = ','.join(n)
        if (data[i]['hidden_entity_sentence'] == None):
            rwes.append(data[i]['review_without_extracted_sentences']['review'])
            p, n = Extract_Attributes(rwes[i], atr_thres_pos, atr_thres_neg, global_pos_list, global_neg_list)
            temp = data[i]['review_without_extracted_sentences']
            temp['Positive_attributes'] = ','.join(p)
            temp['Negative_attributes'] = ','.join(n)

        else:
            for j in range(len(data[i]['hidden_entity_sentence'])):
                hes.append(data[i]['hidden_entity_sentence'][j]['hidden_entity_sentence'])
                for k in range(len(hes)):
                    p, n = Extract_Attributes(hes[k], atr_thres_pos, atr_thres_neg, global_pos_list, global_neg_list)
                    temp = data[i]['hidden_entity_sentence'][j]
                    temp['Positive_attributes'] = ','.join(p)
                    temp['Negative_attributes'] = ','.join(n)
                    # heslen.append(len(data[i]['hidden_entity_sentence']))
            rwes.append(data[i]['review_without_extracted_sentences']['review_without_hidden_entity_sentence_final'])
            p, n = Extract_Attributes(rwes[i], atr_thres_pos, atr_thres_neg, global_pos_list, global_neg_list)
            temp = data[i]['review_without_extracted_sentences']
            temp['Positive_attributes'] = ','.join(p)
            temp['Negative_attributes'] = ','.join(n)

    returnValue = []
    for n in data:
        new_obj = n
        temp_hidden_entity = []
        if new_obj['hidden_entity_sentence'] is None:
            temp_hidden_entity = new_obj['hidden_entity_sentence']
        else:
            for j in new_obj['hidden_entity_sentence']:
                new_entity_obj = j
                pos_attr = list((j['Positive_attributes']).split(",")) if j['Positive_attributes'] != "" else []
                neg_attr = list((j['Negative_attributes']).split(",")) if j['Negative_attributes'] != "" else []
                if(len(neg_attr)>len(pos_attr)):
                    new_entity_obj['hidden_entity_sentence_predicted_sentiment_final']['sentiment'] = "Negative"
                temp_hidden_entity.append(new_entity_obj)
        new_obj['hidden_entity_sentence'] = temp_hidden_entity

        origin_sentence_pos_attr = list((new_obj['original_review']['Positive_attributes']).split(",")) if new_obj['original_review']['Positive_attributes'] != "" else []
        origin_sentence_neg_attr = list((new_obj['original_review']['Negative_attributes']).split(",")) if new_obj['original_review']['Negative_attributes'] != "" else []
        if(len(origin_sentence_neg_attr)>len(origin_sentence_pos_attr)):
            new_obj['original_review']['review_predicted_sentiment_final']['sentiment'] = "Negative"
        
        withoutentity_sentence_pos_attr = list((new_obj['review_without_extracted_sentences']['Positive_attributes']).split(",")) if new_obj['review_without_extracted_sentences']['Positive_attributes'] != "" else []
        withoutentity_sentence_neg_attr = list((new_obj['review_without_extracted_sentences']['Negative_attributes']).split(",")) if new_obj['review_without_extracted_sentences']['Negative_attributes'] != "" else []
        if(len(withoutentity_sentence_neg_attr)>len(withoutentity_sentence_pos_attr)):
            if 'review_without_hidden_entity_sentence_predicted_sentiment_final' in (new_obj['review_without_extracted_sentences']).keys():
                new_obj['review_without_extracted_sentences']['review_without_hidden_entity_sentence_predicted_sentiment_final']['sentiment'] = "Negative"
            if 'review_predicted_sentiment_final' in (new_obj['review_without_extracted_sentences']).keys():
                new_obj['review_without_extracted_sentences']['review_predicted_sentiment_final']['sentiment'] = "Negative"

        returnValue.append(new_obj)

    return returnValue


# The below differs the former Extract_Attributes with the positive and negative threshold
def Extract_Attributes_pos_neg_list(review):
    pos_adj = []
    neg_adj = []
    adj_list = Extract_Adjectives(review)
    for adj in adj_list:
        zen = TextBlob(adj)
        for sentence in zen.sentences:
            if sentence.sentiment.polarity >= 0:
                score = sentence.sentiment.polarity
                score = '%.2f'%score
                adj = adj + "|" + str(score)
                pos_adj.append(adj)
            elif sentence.sentiment.polarity < 0:
                score = sentence.sentiment.polarity
                score = '%.2f'%score
                adj = adj + "|" + str(score)
                neg_adj.append(adj)
    return pos_adj, neg_adj


# A new api that generates the positive and negative list
def generate_pos_neg_list(data_json, file):
    df_entity_specific = jsontodf(data_json, file)
    pos_list = []
    neg_list = []
    for i in range(len(df_entity_specific.index)):
        print(i)
        p, n = Extract_Attributes_pos_neg_list(df_entity_specific['Comment'][i])
        if len(p) != 0:
            for j in range(len(p)):
                pos_list.append(p[j])
        if len(n) != 0:
            for k in range(len(n)):
                neg_list.append(n[k])

    x = np.array(pos_list)
    y = np.array(neg_list)
    print(type(np.unique(x)))
    print(np.unique(x))
    print(len(np.unique(x)))
    positive = list(np.unique(x))
    negative = list(np.unique(y))
    print(("Before pos:{} neg:{}").format(len(pos_list), len(neg_list)))
    print(("After pos:{} neg:{}").format(len(positive), len(negative)))

    msg = {"Positive_list": positive, "Negative_list": negative}

    # msg = json.dumps(msg)
    return msg


def extract_hidden_entity_list(review):
    extracted_sentences = {}
    review_without_hidden_entity = []
    entity_list = []
    sentence_count = 0
    for sentence in sent_tokenize(review):
        sentence_count += 1
        doc = nlp(sentence)
        for token in doc:
            #print(token, token.dep_)
            # print("********")
            if str(token.dep_) == 'nsubj' or str(token.dep_) == 'nsubjpass':
                entity_list.append(token.text)
    #entity_list = ','.join(entity_list)
    return entity_list


def generate_entity_list(data_json,file):
    entity_list=[]
    listofentity=[]
    hipen_list =[]
    df = jsontodf(data_json, file)
    for i, row in df.iterrows():
        print(i)
        list = extract_hidden_entity_list(row['Comment'].lower())
        for i in range (len(list)):
            entity_list.append(list[i])

    x = np.array(entity_list)
    entitylist = np.unique(x)
    print(entitylist)
    for i in range(len(entitylist)):
        if entitylist[i].isalpha():
            listofentity.append(entitylist[i])


    msg = {"entity list:": listofentity}
    return msg

def searchHiddenEntity(df,  list_of_entity_name):
    result = []
    for i, row in df.iterrows():
        neu_cor = pronoun_resolution(row)
        extracted_sentences, review_without_hidden_entity = extract_hidden_entities(neu_cor[0], list_of_entity_name)
        s = extracted_sentences.keys()
        row['Result'] = ', '.join(map(str, s))
        #find missing entities
        x =  [] if isNaN(row['From Compare']) else ((row['From Compare']).lower()).split(",")
        y = row['Result'].split(", ")
        res = list(set(x) - set(y))
        new_res = []
        for n in res:
            if n not in str(row['Result']):
                new_res.append(n)
        row['Missing'] = ', '.join(map(str, new_res))  
        result.append(row)         

    return result

def isNaN(string):
    return string != string

def findEntity(list_of_entity_name=[]):
    if type(list_of_entity_name) != list:
        list_of_entity_name = ast.literal_eval(list_of_entity_name)
    processed_data = []
    review_wo = []
    acceptable_dep = ['pobj', 'dobj', 'iobj', 'ROOT', 'compound', 'conj']
    df = pd.read_excel('Unextrcted Entity.xlsx', encoding='utf-8')
    commend_data = df['Comments']
    provider_data = df['Provider']
    sentence_count = 1
    for row_data in commend_data:
        sentence_count += 1
        extracted_sentences = {}
        for sentence in sent_tokenize(row_data):
            doc = nlp(sentence)
            i = 0
            for token in doc:
                i = i+1
                if str(token.dep_) == 'nsubj' or str(token.dep_) == 'nsubjpass' or (( ((token.pos_)=='NOUN') or ((token.pos_)=='PROPN') ) and ((token.dep_) in acceptable_dep)):
                    token_index = token.i
                    prev_token = doc[token_index - 1]
                    s_prev_token = None
                    s_next_token = None
                    try:
                        s_prev_token =  (doc[token_index -2]) if (doc[token_index -2]) else None
                        s_next_token = (doc[token_index +2]) if (doc[token_index +2]) else None
                    except:
                        print(i)
                    if(i < len(doc)):
                        next_token = doc[token_index + 1]
                        if (prev_token.text.lower() in list_of_entity_name and token.text.lower() in list_of_entity_name) and (next_token.text.lower() in list_of_entity_name and token.text.lower() in list_of_entity_name):
                            word = prev_token.text + " " + token.text + " " + next_token.text
                            if ((s_prev_token) and s_prev_token.text.lower() in list_of_entity_name):
                                word = s_prev_token.text + " " + word
                            extracted_sentences[word] = sentence
                            extracted_sentences['dep_'] = token.dep_
                        elif prev_token.text.lower() in list_of_entity_name and token.text.lower() in list_of_entity_name:
                            word = prev_token.text + " " + token.text
                            if ((s_prev_token) and s_prev_token.text.lower() in list_of_entity_name):
                                word = s_prev_token.text + " " + word
                            extracted_sentences[word] = sentence
                            extracted_sentences['dep_'] = token.dep_
                        elif token.text.lower() in list_of_entity_name and next_token.text.lower() in list_of_entity_name:
                            word = token.text + " " + next_token.text
                            if ((s_next_token) and s_next_token.text.lower() in list_of_entity_name):
                                word =  word + " " + s_next_token.text
                            extracted_sentences[word] = sentence
                            extracted_sentences['dep_'] = token.dep_
                        if prev_token.text.lower() not in list_of_entity_name and next_token.text.lower() not in list_of_entity_name:
                            if token.text.lower() in list_of_entity_name:
                                extracted_sentences[token.text] = sentence
                                extracted_sentences['dep_'] = token.dep_
                        if prev_token.text:
                            match_token = prev_token.text + " " +token.text
                            if (match_token.lower() in list_of_entity_name):
                                extracted_sentences[match_token] = sentence
                                extracted_sentences['dep_'] = token.dep_

                    else:
                        if prev_token.text.lower() in list_of_entity_name and token.text.lower() in list_of_entity_name:
                            word = prev_token.text + " " + token.text
                            if ((s_prev_token) and s_prev_token.text.lower() in list_of_entity_name):
                                word = s_prev_token.text + " " + word
                            extracted_sentences[word] = sentence
                            extracted_sentences['dep_'] = token.dep_
                        if prev_token.text.lower() not in list_of_entity_name and token.text.lower() in list_of_entity_name:
                            extracted_sentences[token.text] = sentence
                            extracted_sentences['dep_'] = token.dep_
                        if prev_token.text:
                            match_token = prev_token.text + " " +token.text
                            if (match_token.lower() in list_of_entity_name):
                                extracted_sentences[match_token] = sentence
                                extracted_sentences['dep_'] = token.dep_
        new_obj = {}
        new_obj['id'] = sentence_count
        try:
            new_obj['provider'] = provider_data[sentence_count-2]
        except:
            print("unbound exception")
        new_obj['extracted_sentence'] = extracted_sentences
        new_obj['original_sentence'] = row_data
        processed_data.append(new_obj)
    returnData = {}
    returnData['result'] = processed_data
    returnData['count_total_record'] = len(processed_data)
    predicted_count = 0
    unPredeicted_count = 0
    for row in processed_data:
        if not row['extracted_sentence']:
            unPredeicted_count = unPredeicted_count + 1
        else:
            predicted_count = predicted_count + 1
    returnData['count_extracted_data'] = predicted_count
    returnData['count_unextracted_data'] = unPredeicted_count
    returnData['extracted_percentage'] = (predicted_count/len(processed_data)) * 100
    return returnData

if __name__ == "__main__":
    atr_thres_pos = 0
    atr_thres_neg = 0.1
    global_pos_list = ["able", "absolute", "absolutely", "abundant", "accessible", "accurate", "adept", "adequate",
                       "adorable",
                       "advanced", "affable", "affirmative", "alive", "amazing", "amenable", "amusing", "appealing",
                       "appreciated", "appreciative", "appropriate", "apt", "astonishing", "astounding", "astute",
                       "attendant",
                       "attentive", "attractive", "authentic", "autonomous", "available", "aware", "awesome",
                       "beautiful",
                       "best", "better", "bright", "brilliant", "calm", "candid", "capable", "certain", "charismatic",
                       "charming", "cheerful", "cheery", "classic", "classy", "clean", "clear", "clever", "coherent",
                       "comfortable", "comic", "competent", "complete", "complimentary", "comprehensible", "concise",
                       "concrete", "confident", "considerable", "consistent", "consummate", "contemporary",
                       "convincing",
                       "cool", "courteous", "creative", "credible", "cultural", "cute", "decent", "deft", "delightful",
                       "deserving", "detailed", "direct", "early", "easy", "economic", "educational", "effective",
                       "endearing",
                       "energetic", "engaging", "enjoyable", "enlightening", "entertaining", "enthusiastic", "epic",
                       "ethical",
                       "exact", "excellent", "exceptional", "excited", "exciting", "experienced", "exquisite",
                       "extraordinary",
                       "Efficient", "Empathetic", "efficient", "empathetic", "fabulous", "fair", "familiar",
                       "fantastic",
                       "fast", "favorite", "fine", "fit", "fitting", "flawless", "fresh", "friendly", "fun", "funny",
                       "gentle",
                       "genuine", "gifted", "glad", "golden", "good", "grand", "great", "greatest", "handsome", "happy",
                       "healthy", "hilarious", "honest", "humorous", "ideal", "impeccable", "important", "impressive",
                       "incomparable", "incredible", "inspirational", "inspiring", "intellectual", "intelligent",
                       "intense",
                       "interesting", "intimate", "inventive", "kind", "kindly", "Knowledgeable", "knowledgeable",
                       "likable",
                       "Listened", "listened", "logical", "lovely", "loving", "loyal", "magnificent", "mannerly",
                       "marvelous",
                       "mature", "meaningful", "memorable", "mild", "modern", "natural", "never hurried",
                       "never judgmental",
                       "never long", "never memorial", "never rushed", "never avaible", "never awful", "never bad",
                       "never close", "never confused", "never critical", "never defensive", "never disappointed",
                       "never disrespectful", "never dull", "nice", "notable", "never efficient", "never embarrased",
                       "never excessive", "never exemplary", "never expensive", "never female", "never frightened",
                       "never hurried", "never judged", "never judgmental", "never late", "never less", "never long",
                       "never male", "never memorial", "never needless", "never negative", "never other",
                       "never overcrowded",
                       "never poor", "never pushy", "never random", "never rude", "never rush", "never rushed",
                       "never rushed!she", "never rusher", "never same", "never secondary", "never serious",
                       "never sick",
                       "never stupid", "never such", "never thorough", "never unanswered", "never uncomfortable",
                       "never unnecessary", "never urgent", "never worried", "never wrong", "new", "new.i", "newborn",
                       "newer",
                       "newest", "next", "nice", "niceband", "nicedoctor", "nicer", "nicest", "nicotine", "nitrous",
                       "no",
                       "no asinine", "no bad", "no big", "no cervical", "no frustrating", "no insignificant",
                       "no little",
                       "no long", "no longer", "no medical", "no minor", "no necessary", "no next", "no sick",
                       "no silly",
                       "no small", "no stupid", "no triple", "no trivial", "no uninformed", "no worried", "noligable",
                       "non",
                       "nonexistent", "nonjudgemental", "nonjudgmental", "nonreplaceable", "nonsense", "nontraditional",
                       "normal", "not -", "not 1st", "not abrasive", "not acceptable", "not activate", "not active",
                       "not actual", "not additional", "not adverse", "not afraid", "not aggressive", "not alarmist",
                       "not allergic", "not alone", "not alternate", "not amercian", "not ample", "not annual",
                       "not anti",
                       "not antibiotic", "not anxious", "not arbitrary", "not arrogant", "not autoimmune",
                       "not average",
                       "not awkward", "not back", "not bad", "not basic", "not bedside", "not big", "not brisk",
                       "not caring",
                       "not catastrophic", "not clinical", "not cold", "not common", "not compassionate",
                       "not competant",
                       "not complex", "not comprehensive", "not concerned", "not constructive", "not contagious",
                       "not convenient", "not convinced", "not correct", "not crazy", "not critical", "not cumbersome",
                       "not current", "not curt", "not dandruff", "not dangerous", "not desirable", "not diabetic",
                       "not different", "not difficult", "not dire", "not dirty", "not disappointed", "not dismissive",
                       "not dissatisfied", "not drs", "not due", "not eager", "not efficient", "not ego",
                       "not egotistical",
                       "not embarrassed", "not empathetic", "not emphatic", "not enough", "not entire", "not expensive",
                       "not extra", "not fake", "not false", "not fare", "not female", "not few", "not fluid",
                       "not flush",
                       "not focused", "not forceful", "not further", "not generic", "not geriatric", "not glowing",
                       "not grateful", "not grueling", "not happier", "not hard", "not helpful", "not hesitant",
                       "not highest",
                       "not hippocratic", "not holistic", "not hurried", "not ibuprofen", "not ignorant",
                       "not immediate",
                       "not impatient", "not inclined", "not informative", "not informed", "not intermittent",
                       "not interpersonal", "not intimidated", "not intimidating", "not intrusive", "not involved",
                       "not judged", "not judgemental", "not judgmental", "not keen", "not knowledgeable", "not last",
                       "not layman", "not least", "not less", "not likely", "not listened", "not little", "not long",
                       "not longer", "not mean", "not medical", "not memmorial", "not memorial", "not minimal",
                       "not minor",
                       "not nagging", "not nameless", "not necesary", "not necessary", "not negative", "not nervous",
                       "not neurological", "not next", "not nicer", "not nutrician", "not occupied", "not only",
                       "not open",
                       "not optimal", "not organized", "not other", "not ovarian", "not overbearing", "not overdue",
                       "not overwhelmed", "not painful", "not past", "not patient", "not patronizing", "not pediatric",
                       "not personable", "not personal", "not personalible", "not physical", "not pink", "not polite",
                       "not possible", "not pre", "not precsribe", "not prepared", "not prescribed", "not present",
                       "not pressing", "not pressured", "not pretentious", "not preventive", "not previous",
                       "not prior",
                       "not private", "not proactive", "not prompt", "not psychiatric", "not puahy", "not public",
                       "not pushy",
                       "not qualified", "not reactive", "not reassuring", "not receptive", "not regular", "not related",
                       "not reluctant", "not reticent", "not rheumatoid", "not rude", "not rush", "not rushed",
                       "not same",
                       "not senior", "not serious", "not severe", "not short", "not sick", "not sicker", "not sickness",
                       "not side", "not single", "not slow", "not softer", "not somber", "not sorry", "not specific",
                       "not standard", "not sterile", "not stiff", "not stressful", "not stupid", "not such",
                       "not sufficient",
                       "not sugarcoat", "not supplant", "not technical", "not tense", "not terrible", "not thankful",
                       "not third", "not thorough", "not time.negative", "not tolerable", "not touchy", "not typical",
                       "not ugly", "not ultimate-", "not uncomfortable", "not undivided", "not unhappy",
                       "not unnecessary",
                       "not unprofessional", "not unrealistic", "not unsatisfactory", "not unsatisfied", "not urgent",
                       "not usual", "not valid", "not verbal", "not wart", "not weekly", "not weird", "not well",
                       "not wince",
                       "not worried", "not worse", "not wrong", "noteworthy", "original", "outstanding", "peaceful",
                       "perfect",
                       "phenomenal", "pleasant|0.74","pleasantly|0.8",  "pleased", "popular", "positive", "powerful", "precious", "precise",
                       "pretty",
                       "Personable", "personable", "priceless", "professional", "profound", "pure", "quick", "ready",
                       "realistic", "reasonable", "refreshing", "relevant", "remarkable", "resourceful", "respectable",
                       "respectful", "responsible", "right", "safe", "satisfying", "seamless", "seasoned", "secure",
                       "sensitive", "sexy", "significant", "sincere", "skilled", "smart", "smile", "smooth", "social",
                       "soft",
                       "special", "spectacular", "spirited", "splendid", "steady", "stellar", "straightforward",
                       "strong",
                       "success", "successful", "suitable", "super", "superb", "superior", "supportive", "sweet",
                       "sympathetic", "talented", "thoughtful", "thrilled", "tidy", "touching", "tremendous",
                       "truthful",
                       "uncommon", "unforgettable", "unique", "useful", "vibrant", "vital", "warm", "welcome",
                       "willing",
                       "wise", "witty", "wonderful", "worth", "worthwhile", "worthy", "young"]
    
    global_neg_list = ["abrupt", "addicted", "afraid", "aged", "angrily", "angry", "annoyed", "annoying", "anxious",
                       "arbitrary", "atrocious", "average", "arrogant", "awful", "awkward", "bad", "bored", "boring",
                       "broken", "casual",
                       "cluelessness", "cold", "complicated", "confused", "confusing", "corrupt", "crazy", "cruel",
                       "dangerous", "dark", "desperate", "difficult", "dirty", "disappointed", "disappointing",
                       "disgusting",
                       "dishonest", "distant", "disturbing", "dramatic", "dreadful", "dull", "dumb", "empty",
                       "erroneous",
                       "evil", "excessive", "expensive", "fake", "false", "fearful", "filthy", "flat", "forgetful",
                       "frightening", "frustrated", "frustrating", "green", "harsh", "horrible|0.4","horriblely|0.7", "ill", "illegal",
                       "impatient",
                       "impossible", "incompetent", "inexperienced", "insecure", "insulting", "irrelevant", "late",
                       "lazy",
                       "limited", "lousy", "mad", "mean", "mediocre", "messy", "miserable", "narrow", "needless",
                       "negative",
                       "never able", "never better", "never certain", "never comfortable", "never direct", "never easy",
                       "never effective", "never excellent", "never fun", "never great", "never more", "never much",
                       "never pleasant", "never pleased", "never primary", "never quick", "no better", "no busy",
                       "no many",
                       "no more", "no much", "no new", "not able", "not accessible", "not accurate", "not adequate",
                       "not alive", "not appreciative", "not appropriate", "not attentive", "not available",
                       "not aware",
                       "not best", "not better", "not bright", "not busy", "not certain", "not cheap", "not clean",
                       "not clear", "not comfortable", "not competent", "not complete", "not confident",
                       "not consummate",
                       "not convincing", "not cool", "not detailed", "not direct", "not early", "not easy",
                       "not effective",
                       "not excellent", "not exceptional", "not fair", "not familiar", "not fantastic", "not favorite",
                       "not first", "not friendly", "not full", "not fun", "not general", "not gentle", "not genuine",
                       "not good", "not great", "not greatest", "not happy", "not high", "not honest", "not hot",
                       "not important", "not impressed", "not intelligent", "not interested", "not kind", "not loud",
                       "not major", "not many", "not more", "not most", "not much", "not natural", "not new",
                       "not nice",
                       "not normal", "not old", "not original", "not outstanding", "not own", "not particular",
                       "not perfect",
                       "not pleasant", "not pleased", "not positive", "not primary", "not professional", "not proud",
                       "not quick", "not rare", "not ready", "not real", "not remarkable", "not respectful",
                       "not right",
                       "not satisfied", "not satisfying", "not secure", "not sensitive", "not sincere", "not social",
                       "not strong", "not successful", "not sure", "not surprised", "not surprising", "not sympathetic",
                       "not thrilled", "not true", "not useful", "not warm", "not whole", "not willing",
                       "not wonderful",
                       "not worth", "not worthy", "not young", "ordinary", "outraged", "painful", "pathetic", "plain",
                       "pointless", "poor", "pretentious", "questionable", "repellent", "repetitive", "ridiculous",
                       "robotic",
                       "rough", "rude", "sad", "scary", "selfish", "shady", "shaky", "shallow", "shocked", "shocking",
                       "shy",
                       "sick", "sickly", "silly", "simplistic", "sleepy", "sloppy", "slow", "sour", "stiff", "strange",
                       "stupid", "tedious", "tense", "terrible", "tired", "tough", "ugly", "unanswered", "unbelievable",
                       "uncomfortable", "unexplained", "unfortunate", "unhappy", "unhealthy", "unimportant", "unknown",
                       "unlikely", "unnecessary", "unpleasant", "unpredictable", "unprofessional", "unrealistic",
                       "useless",
                       "usual", "vague", "weak", "weird", "worse", "worst", "wrong"]
    #file = "C:/Users/Asus/Desktop/dual subject conjunction.xlsx"

    file = ""
    # data_json = '''[{"Provider": "Dr.Ewald, MD","Review": " The nurses who made the appointment checked me in, and took bloodwork were absolutely fantastic. The doctor, however, gave me extreme difficulty even trying to ask for an STD check to be performed on my bloodwork. He would ask me questions about past doctors Ive been to, but gave me an extremely difficult time with my responses. I will never come see this doctor again, and I will certainly not recommend him to family or friends. "}]'''
    data_json = ''' [{"Provider": "Dr. Ali K.","Review": "Dr. K  is very impressive, we really have confidence in him and his assistant was helpful. We really appreciated his taking time to really talk to John about his life and professional occupation. We came away feeling like we have a new friend...Thank you, Alice and John Wallace"}]'''
    

    #data_json = {}
    vectorizer_filename = "F:/OPTISOL/Project/Repugen/Repugen/upload/vectorizer.pkl"
    classifier_filename = "F:/OPTISOL/Project/Repugen/Repugen/upload/classifier_model.pkl"
    list_of_entity_name = ['technicians', 'nurses', 'facility', 'assistant', 'desk', 'rep', 'lpn', 'office', 'rn',
                           'establishment', 'provider', 'specialists', 'employee', 'nurse', 'np', 'receptionist', 'pa',
                           'employees', 'staff', 'attendant', 'practitioner', 'reception', 'technician', 'offices',
                           'facilities', 'assistants', 'providers', 'specialist', 'staffs', 'attendants',
                           'practitioners', 'lab']

    max_pos_threshold = 0.5
    max_neg_threshold = -0.5

    var = ExtractHidden_Score_Generate_Attributes(data_json, file, classifier_filename, vectorizer_filename,
                                                  atr_thres_pos, atr_thres_neg, global_pos_list, global_neg_list,
                                                  list_of_entity_name,max_pos_threshold,max_neg_threshold)
    # var = generate_pos_neg_list(data_json,file)
    # var = extracting_hidden_entities_and_reviews_and_sentiment_prediction(file, vectorizer_filename, classifier_filename, data_json,list_of_entity_name)
    var1 = json.dumps(var)
    # print(var)
    print("output: ",var1)







