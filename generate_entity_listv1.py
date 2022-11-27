import sys
import numpy as np
import spacy
import os
from tqdm import tqdm
import json
import pandas as pd
import warnings
from datetime import date
warnings.filterwarnings('ignore')
from pandas.io.json import json_normalize
from nltk.tokenize import sent_tokenize
nlp = spacy.load("en_core_web_sm")

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

def extract_hidden_entity_list(review):
    extracted_sentences = {}
    review_without_hidden_entity = []
    entity_list = []
    sentence_count = 0
    for sentence in sent_tokenize(review):
        sentence_count += 1
        try:
            doc = nlp(sentence)
        except Exception:
            continue
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
    for i, row in tqdm(df.iterrows(),total=df.shape[0]):
        #print(i)
        list = extract_hidden_entity_list(row['Comment'].lower())
        for i in range (len(list)):
            entity_list.append(list[i])

    x = np.array(entity_list)
    entitylist = np.unique(x)
    #print(entitylist)
    #for i in tqdm(range(int(9e6))):
    for i in range(len(entitylist)):
        if entitylist[i].isalpha():
            listofentity.append(entitylist[i])
    msg = {"entity list:": listofentity}
    return msg

if __name__ == "__main__":
    file = sys.argv[1]
    data_json = {}
    var = generate_entity_list(data_json, file)
    today = date.today()
    d1 = today.strftime("%d.%m.%Y")
    if not os.path.exists('./out'):
        os.mkdir("./out")
    with open('./out/'+'entity_list_' + d1 + '.json', 'w') as outfile:
        data = var
        json.dump(data, outfile)

