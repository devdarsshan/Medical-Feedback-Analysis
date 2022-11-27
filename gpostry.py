import sys
import re
import numpy as np
import spacy
import os
import json
import neuralcoref
from tqdm import tqdm
import pandas as pd
import warnings
from datetime import date
warnings.filterwarnings('ignore')
from pandas.io.json import json_normalize
from nltk.tokenize import sent_tokenize
from datetime import datetime
from textblob import TextBlob
now = datetime.now()
nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)



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
def Extract_Adjectives(review):
    keyword_lookup = []
    flag = 0
    for sentence in sent_tokenize(review):
        try:
            doc = nlp(sentence)
        except Exception:
            continue
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

def Extract_Attributes_pos_neg_list(review):
    pos_adj = []
    neg_adj = []

    adj_list = Extract_Adjectives(review)

    for adj in adj_list:
        if "\\" not in adj:
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

def generate_pos_neg_list(data_json, file):
    df_entity_specific = jsontodf(data_json, file)
    pos_list = []
    neg_list = []
    for i in tqdm(range(len(df_entity_specific.index))):
        p, n = Extract_Attributes_pos_neg_list(df_entity_specific['Comment'][i])
        if len(p) != 0:
            for j in range(len(p)):
                    pos_list.append(p[j])
        if len(n) != 0:
            for k in range(len(n)):
                    neg_list.append(n[k])

    x = np.array(pos_list)
    y = np.array(neg_list)

    positive = list(np.unique(x))
    negative = list(np.unique(y))


    msg = {"Positive_list": positive, "Negative_list": negative}
    return msg

if __name__ == "__main__":
    file = sys.argv[1]
    data_json = {}
    var = generate_pos_neg_list(data_json, file)
    today = date.today()

    d1 = today.strftime("%d.%m.%Y")
    if not os.path.exists('./out'):
        os.mkdir("./out")
    with open('./out/pos_neg_list_' + d1 + '.json', 'w') as outfile:
        data = var
        json.dump(data, outfile)
