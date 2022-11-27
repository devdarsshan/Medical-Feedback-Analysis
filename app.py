
from flask import Flask, request, jsonify
import pandas as pd
import os
import time
from logging.handlers import  RotatingFileHandler
import logging
from datetime import datetime
import traceback
import app_fin as api
from time import strftime
from logging.handlers import TimedRotatingFileHandler
import json


#log_path = os.getcwd()
app = Flask(__name__)
log_file = r"./logs/app.log"
#Need to change the log file at the clients machine
#handler = RotatingFileHandler( log_file, maxBytes=50*1024*1024, backupCount=50)
handler = TimedRotatingFileHandler(log_file, when='D', interval=1,backupCount=10, encoding=None, delay=False,utc=False, atTime=None)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
logger.addHandler(handler)

DEFAULT_THRESHOLD_CONST = 0.51
DEFAULT_NEG_THRESHOLD_CONST = 0.4852


def check_parameters(data):
    data_dict = data.to_dict(flat=True)
    if 'data_json' not in data_dict:
        data_dict['data_json'] = {}
    if 'vectorizer_filename' not in data_dict:
        data_dict['vectorizer_filename'] = "vectorizer2020-02-03 06:02:10.629805.pkl"
    if 'classifier_filename' not in data_dict:
        data_dict['classifier_filename'] = "classifier_model2020-02-03 06:02:10.629805.pkl"
    if 'Bigram_Boolean' not in data_dict:
        data_dict['Bigram_Boolean'] = "True"
    if 'top_x_attributes' not in data_dict:
        data_dict['top_x_attributes'] = "10"
    if 'atr_thres_pos' not in data_dict:
        data_dict['atr_thres_pos'] = "0"
    if 'atr_thres_neg' not in data_dict:
        data_dict['atr_thres_neg'] = "0"
    if 'global_pos_list' not in data_dict:
        data_dict['global_pos_list'] = ["able", "absolute", "absolutely", "abundant", "accessible", "accurate", "adept", "adequate", "adorable", "advanced", "affable", "affirmative", "alive", "amazing", "amenable", "amusing", "appealing", "appreciated", "appreciative", "appropriate", "apt", "astonishing", "astounding", "astute", "attendant", "attentive", "attractive", "authentic", "autonomous", "available", "aware", "awesome", "beautiful", "best", "better", "bright", "brilliant", "calm", "candid", "capable", "certain", "charismatic", "charming", "cheerful", "cheery", "classic", "classy", "clean", "clear", "clever", "coherent", "comfortable", "comic", "competent", "complete", "complimentary", "comprehensible", "concise", "concrete", "confident", "considerable", "consistent", "consummate", "contemporary", "convincing", "cool", "courteous", "creative", "credible", "cultural", "cute", "decent", "deft", "delightful", "deserving", "detailed", "direct", "early", "easy", "economic", "educational", "effective", "endearing", "energetic", "engaging", "enjoyable", "enlightening", "entertaining", "enthusiastic", "epic", "ethical", "exact", "excellent", "exceptional", "excited", "exciting", "experienced", "exquisite", "extraordinary","Efficient","Empathetic", "efficient","empathetic", "fabulous", "fair", "familiar", "fantastic", "fast", "favorite", "fine", "fit", "fitting", "flawless", "fresh", "friendly", "fun", "funny", "gentle", "genuine", "gifted", "glad", "golden", "good", "grand", "great", "greatest", "handsome", "happy", "healthy", "hilarious", "honest", "humorous", "ideal", "impeccable", "important", "impressive", "incomparable", "incredible", "inspirational", "inspiring", "intellectual", "intelligent", "intense", "interesting", "intimate", "inventive", "kind", "kindly", "Knowledgeable","knowledgeable","likable", "Listened","listened","logical", "lovely", "loving", "loyal", "magnificent", "mannerly", "marvelous", "mature", "meaningful", "memorable", "mild", "modern", "natural","never hurried", "never judgmental", "never long", "never memorial", "never rushed", "never avaible", "never awful", "never bad", "never close", "never confused", "never critical", "never defensive", "never disappointed", "never disrespectful", "never dull","nice", "notable", "never efficient", "never embarrased", "never excessive", "never exemplary", "never expensive", "never female", "never frightened", "never hurried", "never judged", "never judgmental", "never late", "never less", "never long", "never male", "never memorial", "never needless", "never negative", "never other", "never overcrowded", "never poor", "never pushy", "never random", "never rude", "never rush", "never rushed", "never rushed!she", "never rusher", "never same", "never secondary", "never serious", "never sick", "never stupid", "never such", "never thorough", "never unanswered", "never uncomfortable", "never unnecessary", "never urgent", "never worried", "never wrong", "new", "new.i", "newborn", "newer", "newest", "next", "nice", "niceband", "nicedoctor", "nicer", "nicest", "nicotine", "nitrous", "no", "no asinine", "no bad", "no big", "no cervical", "no frustrating", "no insignificant", "no little", "no long", "no longer", "no medical", "no minor", "no necessary", "no next", "no sick", "no silly", "no small", "no stupid", "no triple", "no trivial", "no uninformed", "no worried", "noligable", "non", "nonexistent", "nonjudgemental", "nonjudgmental", "nonreplaceable", "nonsense", "nontraditional", "normal", "not -", "not 1st", "not abrasive", "not acceptable", "not activate", "not active", "not actual", "not additional", "not adverse", "not afraid", "not aggressive", "not alarmist", "not allergic", "not alone", "not alternate", "not amercian", "not ample", "not annual", "not anti", "not antibiotic", "not anxious", "not arbitrary", "not arrogant", "not autoimmune", "not average", "not awkward", "not back", "not bad", "not basic", "not bedside", "not big", "not brisk", "not caring", "not catastrophic", "not clinical", "not cold", "not common", "not compassionate", "not competant", "not complex", "not comprehensive", "not concerned", "not constructive", "not contagious", "not convenient", "not convinced", "not correct", "not crazy", "not critical", "not cumbersome", "not current", "not curt", "not dandruff", "not dangerous", "not desirable", "not diabetic", "not different", "not difficult", "not dire", "not dirty", "not disappointed", "not dismissive", "not dissatisfied", "not drs", "not due", "not eager", "not efficient", "not ego", "not egotistical", "not embarrassed", "not empathetic", "not emphatic", "not enough", "not entire", "not expensive", "not extra", "not fake", "not false", "not fare", "not female", "not few", "not fluid", "not flush", "not focused", "not forceful", "not further", "not generic", "not geriatric", "not glowing", "not grateful", "not grueling", "not happier", "not hard", "not helpful", "not hesitant", "not highest", "not hippocratic", "not holistic", "not hurried", "not ibuprofen", "not ignorant", "not immediate", "not impatient", "not inclined", "not informative", "not informed", "not intermittent", "not interpersonal", "not intimidated", "not intimidating", "not intrusive", "not involved", "not judged", "not judgemental", "not judgmental", "not keen", "not knowledgeable", "not last", "not layman", "not least", "not less", "not likely", "not listened", "not little", "not long", "not longer", "not mean", "not medical", "not memmorial", "not memorial", "not minimal", "not minor", "not nagging", "not nameless", "not necesary", "not necessary", "not negative", "not nervous", "not neurological", "not next", "not nicer", "not nutrician", "not occupied", "not only", "not open", "not optimal", "not organized", "not other", "not ovarian", "not overbearing", "not overdue", "not overwhelmed", "not painful", "not past", "not patient", "not patronizing", "not pediatric", "not personable", "not personal", "not personalible", "not physical", "not pink", "not polite", "not possible", "not pre", "not precsribe", "not prepared", "not prescribed", "not present", "not pressing", "not pressured", "not pretentious", "not preventive", "not previous", "not prior", "not private", "not proactive", "not prompt", "not psychiatric", "not puahy", "not public", "not pushy", "not qualified", "not reactive", "not reassuring", "not receptive", "not regular", "not related", "not reluctant", "not reticent", "not rheumatoid", "not rude", "not rush", "not rushed", "not same", "not senior", "not serious", "not severe", "not short", "not sick", "not sicker", "not sickness", "not side", "not single", "not slow", "not softer", "not somber", "not sorry", "not specific", "not standard", "not sterile", "not stiff", "not stressful", "not stupid", "not such", "not sufficient", "not sugarcoat", "not supplant", "not technical", "not tense", "not terrible", "not thankful", "not third", "not thorough", "not time.negative", "not tolerable", "not touchy", "not typical", "not ugly", "not ultimate-", "not uncomfortable", "not undivided", "not unhappy", "not unnecessary", "not unprofessional", "not unrealistic", "not unsatisfactory", "not unsatisfied", "not urgent", "not usual", "not valid", "not verbal", "not wart", "not weekly", "not weird", "not well", "not wince", "not worried", "not worse", "not wrong", "noteworthy","original", "outstanding", "peaceful", "perfect", "phenomenal", "pleasant", "pleased", "popular", "positive", "powerful", "precious", "precise", "pretty", "Personable","personable", "priceless", "professional", "profound", "pure", "quick", "ready", "realistic", "reasonable", "refreshing", "relevant", "remarkable", "resourceful", "respectable", "respectful", "responsible", "right", "safe", "satisfying", "seamless", "seasoned", "secure", "sensitive", "sexy", "significant", "sincere", "skilled", "smart", "smile", "smooth", "social", "soft", "special", "spectacular", "spirited", "splendid", "steady", "stellar", "straightforward", "strong", "success", "successful", "suitable", "super", "superb", "superior", "supportive", "sweet", "sympathetic", "talented", "thoughtful", "thrilled", "tidy", "touching", "tremendous", "truthful", "uncommon", "unforgettable", "unique", "useful", "vibrant", "vital", "warm", "welcome", "willing", "wise", "witty", "wonderful", "worth", "worthwhile", "worthy", "young"]
    if 'global_neg_list' not in data_dict:
        data_dict['global_neg_list'] = ["abrupt", "addicted", "afraid", "aged", "angrily", "angry", "annoyed", "annoying", "anxious", "arbitrary", "atrocious", "average", "awful", "awkward", "bad", "bored", "boring", "broken", "casual", "cluelessness", "cold", "complicated", "confused", "confusing", "corrupt", "crazy", "cruel", "dangerous", "dark", "desperate", "difficult", "dirty", "disappointed", "disappointing", "disgusting", "dishonest", "distant", "disturbing", "dramatic", "dreadful", "dull", "dumb", "empty", "erroneous", "evil", "excessive", "expensive", "fake", "false", "fearful", "filthy", "flat", "forgetful", "frightening", "frustrated", "frustrating", "green", "harsh", "horrible", "ill", "illegal", "impatient", "impossible", "incompetent", "inexperienced", "insecure", "insulting", "irrelevant", "late", "lazy", "limited", "lousy", "mad", "mean", "mediocre", "messy", "miserable", "narrow", "needless", "negative","never able", "never better", "never certain", "never comfortable", "never direct", "never easy", "never effective", "never excellent", "never fun", "never great", "never more", "never much", "never pleasant", "never pleased", "never primary", "never quick", "no better", "no busy", "no many", "no more", "no much", "no new", "not able", "not accessible", "not accurate", "not adequate", "not alive", "not appreciative", "not appropriate", "not attentive", "not available", "not aware", "not best", "not better", "not bright", "not busy", "not certain", "not cheap", "not clean", "not clear", "not comfortable", "not competent", "not complete", "not confident", "not consummate", "not convincing", "not cool", "not detailed", "not direct", "not early", "not easy", "not effective", "not excellent", "not exceptional", "not fair", "not familiar", "not fantastic", "not favorite", "not first", "not friendly", "not full", "not fun", "not general", "not gentle", "not genuine", "not good", "not great", "not greatest", "not happy", "not high", "not honest", "not hot", "not important", "not impressed", "not intelligent", "not interested", "not kind", "not loud", "not major", "not many", "not more", "not most", "not much", "not natural", "not new", "not nice", "not normal", "not old", "not original", "not outstanding", "not own", "not particular", "not perfect", "not pleasant", "not pleased", "not positive", "not primary", "not professional", "not proud", "not quick", "not rare", "not ready", "not real", "not remarkable", "not respectful", "not right", "not satisfied", "not satisfying", "not secure", "not sensitive", "not sincere", "not social", "not strong", "not successful", "not sure", "not surprised", "not surprising", "not sympathetic", "not thrilled", "not true", "not useful", "not warm", "not whole", "not willing", "not wonderful", "not worth", "not worthy", "not young", "ordinary", "outraged", "painful", "pathetic", "plain", "pointless", "poor", "pretentious", "questionable", "repellent", "repetitive", "ridiculous", "robotic", "rough", "rude", "sad", "scary", "selfish", "shady", "shaky", "shallow", "shocked", "shocking", "shy", "sick", "sickly", "silly", "simplistic", "sleepy","sloppy", "slow", "sour", "stiff", "strange", "stupid", "tedious", "tense", "terrible", "tired", "tough", "ugly", "unanswered", "unbelievable", "uncomfortable", "unexplained", "unfortunate", "unhappy", "unhealthy", "unimportant", "unknown", "unlikely", "unnecessary", "unpleasant", "unpredictable","unprofessional", "unrealistic", "useless", "usual", "vague", "weak", "weird", "worse", "worst", "wrong"]
    if 'list_of_entity_name' not in data_dict:
        data_dict['list_of_entity_name'] = ["practice manager", "manager", "location", "technicians", "nurses", "facility", "assistant", "desk", "rep", "lpn", "office", "rn", "establishment", "provider", "specialists", "employee", "nurse", "np", "receptionist", "pa", "employees", "staff", "attendant", "practitioner", "reception", "technician", "offices", "facilities", "assistants", "providers", "specialist", "staffs", "attendants", "practitioners", "lab", "front", "desk","wait time", "bedside manner", "telemedicine", "teleconference", "teledoc", "no wait"]
    return data_dict


def check_files(files):
    files_dict = files.to_dict(flat=True)
    if 'data_file' not in files_dict:
        files_dict['data_file'] = ""
    return files_dict

@app.route('/training_the_classifier', methods=["POST", "GET"])
def train():
    files = request.files
    files = check_files(files)
    data_file = files['data_file']

    if not data_file:
        return "No file"

    else:
        path = "upload/" + data_file.filename
        data_file.save(path)

        result = api.train_classifier(path)
    return jsonify(result)


@app.route('/extracting_hidden_entities_and_reviews_and_sentiment_prediction', methods=['GET', 'POST'])
def predict():
    data = request.form
    files = request.files
    data = check_parameters(data)
    files = check_files(files)
    data_file = files['data_file']
    vectorizer_file = data['vectorizer_filename']
    # vf = request.args.get('vectorizer_filename',type=str)
    classifier_file = data['classifier_filename']
    data_json = data['data_json']
    list_of_entity_name = data['list_of_entity_name']
    max_threshold = DEFAULT_THRESHOLD_CONST if ('max_threshold' not in data) else data['max_threshold']
    # vectorizer_filename = request.args.get('vectorizer_filename',)
    # return jsonify(vectorizer_file)

    # flag = int(request.form['flag'])

    print(len(data_json))
    # print(flag)

    path_1 = ''
    #  without flag
    if len(data_json) == 0 and not data_file:
        return jsonify('Data file need to be presented in either data file and data_json')

    elif len(data_json) == 0:

        if data_file:
            path_1 = "./upload/" + data_file.filename
            data_file.save(path_1)

    elif len(data_json) > 0 and data_file:
        path_1 = ""

    if vectorizer_file in os.listdir("./upload"):
        path_2 = "./upload/" + vectorizer_file
        # vectorizer_file.save(path_2)
    else:
        return jsonify({"message": "No file vectorizer_file"})

    if classifier_file in os.listdir("./upload"):

        path_3 = "./upload/" + classifier_file
        # classifier_file.save(path_3)
    else:
        return jsonify({"message": "No file classifier_file"})

    print("---------------------------")

    # result = api.extracting_hidden_entities_and_reviews_and_sentiment_prediction(path_1, path_2, path_3,int(flag),data_json)
    result = api.extracting_hidden_entities_and_reviews_and_sentiment_prediction(path_1, path_2, path_3, data_json,list_of_entity_name, max_threshold)
    # obj = {hidden_entity_sentence: result, review_without_hidden_entity: result_review_sans_extracted_sentences}
    return jsonify(result)

@app.route('/generate_pos_neg_list', methods=['GET', 'POST'])
def generate_pos_neg_list():
    data = request.form
    files = request.files
    data = check_parameters(data)
    files = check_files(files)
    data_file = files['data_file']
    data_json = data['data_json']
    print("*****************************************")
    print(len(data_json))
    path_1 = ''
    #  without flag
    if len(data_json) == 0 and not data_file:
        return jsonify('Data file need to be presented in either data file and data_json')

    elif len(data_json) == 0:

        if data_file:
            path_1 = "./upload/" + data_file.filename
            data_file.save(path_1)

    elif len(data_json) > 0 and data_file:
        path_1 = ""

    result = api.generate_pos_neg_list(data_json , path_1)
    # obj = {hidden_entity_sentence: result, review_without_hidden_entity: result_review_sans_extracted_sentences}
    return jsonify(result)

@app.route('/generate_entity_list', methods=['GET', 'POST'])
def generate_entity_list():
    data = request.form
    files = request.files
    data = check_parameters(data)
    files = check_files(files)
    data_file = files['data_file']
    data_json = data['data_json']
    print("*****************************************")
    print(len(data_json))
    path_1 = ''
    #  without flag
    if len(data_json) == 0 and not data_file:
        return jsonify('Data file need to be presented in either data file and data_json')

    elif len(data_json) == 0:

        if data_file:
            path_1 = "./upload/" + data_file.filename
            data_file.save(path_1)

    elif len(data_json) > 0 and data_file:
        path_1 = ""

    result = api.generate_entity_list(data_json , path_1)
    # obj = {hidden_entity_sentence: result, review_without_hidden_entity: result_review_sans_extracted_sentences}
    return jsonify(result)

@app.route('/extracthidden_score_generate_attributes', methods=['GET', 'POST'])
def extract_hidden_score_generate_attributes():
    data = request.form
    files = request.files
    data = check_parameters(data)
    files = check_files(files)
    data_file = files['data_file']
    vectorizer_file = data['vectorizer_filename']
    classifier_file = data['classifier_filename']
    atr_thres_pos = data['atr_thres_pos']
    atr_thres_neg = data['atr_thres_neg']
    data_json = data['data_json']
    atr_thres_pos = float(atr_thres_pos)
    atr_thres_neg = float(atr_thres_neg)
    global_pos_list = data['global_pos_list']
    global_neg_list = data['global_neg_list']
    list_of_entity_name = data['list_of_entity_name']

    print("*****************************************")

    path_1 = ''
    #  without flag
    if len(data_json) == 0 and not data_file:
        return jsonify('Data file need to be presented in either data file and data_json')

    elif len(data_json) == 0:

        if data_file:
            path_1 = "./upload/" + data_file.filename
            data_file.save(path_1)

    elif len(data_json) > 0 and data_file:
        path_1 = ""

    if vectorizer_file in os.listdir("./upload"):
        path_2 = "./upload/" + vectorizer_file
        # vectorizer_file.save(path_2)
    else:
        return jsonify({"message": "No file vectorizer_file"})

    if classifier_file in os.listdir("./upload"):

        path_3 = "./upload/" + classifier_file
        # classifier_file.save(path_3)
    else:
        return jsonify({"message": "No file classifier_file"})

    
    max_pos_threshold = DEFAULT_THRESHOLD_CONST if ('max_pos_threshold' not in data) else data['max_pos_threshold']
    max_neg_threshold = DEFAULT_NEG_THRESHOLD_CONST if ('max_neg_threshold' not in data) else data['max_neg_threshold']

    result = api.ExtractHidden_Score_Generate_Attributes(data_json,path_1,path_3, path_2, atr_thres_pos, atr_thres_neg, global_pos_list, global_neg_list,list_of_entity_name,max_pos_threshold, max_neg_threshold)

    return jsonify(result)

@app.route('/generate_common_positive_negative_attributes', methods=['GET', 'POST'])
def generate_common_positive_negative_attributes():
    data = request.form
    files = request.files
    data = check_parameters(data)
    files = check_files(files)
    data_file = files['data_file']
    vectorizer_file = data['vectorizer_filename']
    # vf = request.args.get('vectorizer_filename',type=str)
    classifier_file = data['classifier_filename']
    top_x_attributes = data['top_x_attributes']
    Bigram_Boolean = data['Bigram_Boolean']
    data_json = data['data_json']
    print("*****************************************")
    print(len(data_json))
    # print(flag)

    path_1 = ''
    path = ''
    # without flag
    if len(data_json) == 0 and not data_file:
        return jsonify('Data file need to be presented in either data file and data_json')

    elif len(data_json) == 0:

        if data_file:
            path_1 = "./upload/" + data_file.filename
            data_file.save(path_1)

    elif len(data_json) > 0 and data_file:
        path_1 = ""

    if vectorizer_file in os.listdir("./upload"):
        path_2 = "./upload/" + vectorizer_file
        # vectorizer_file.save(path_2)
    else:
        return "No file vectorizer_file"

    if classifier_file in os.listdir("./upload"):

        path_3 = "./upload/" + classifier_file
        # classifier_file.save(path_3)
    else:
        return "No file classifier_file"
    if not data_file or len(top_x_attributes) == 0:
        result = {"message": "No file"}
        print(result, {"Status": 400})
        # else:
    #     path="./upload/"+data_file.filename
    #     data_file.save(path)

    result = api.generate_common_positive_negative_attributes(path_1, int(top_x_attributes), Bigram_Boolean, path_2,
                                                              path_3, data_json)

    return jsonify(result)



@app.route('/generate_entity_specific_positive_negative_attributes', methods=['GET', 'POST'])
def generate_entity_specific_positive_negative_attributes():
    data = request.form
    files = request.files
    data = check_parameters(data)
    files = check_files(files)
    data_file = files['data_file']
    vectorizer_file = data['vectorizer_filename']
    # vf = request.args.get('vectorizer_filename',type=str)
    classifier_file = data['classifier_filename']
    top_x_attributes = data['top_x_attributes']
    Bigram_Boolean = data['Bigram_Boolean']
    data_json = data['data_json']
    list_of_entity_name = data['list_of_entity_name']
    print("*****************************************")
    print(len(data_json))
    # print(flag)

    path_1 = ''

    if len(data_json) == 0 and not data_file:
        return jsonify('Data file need to be presented in either data file and data_json')

    elif len(data_json) == 0:

        if data_file:
            path_1 = "./upload/" + data_file.filename
            data_file.save(path_1)
            data_json = ""

    elif len(data_json) > 0 and data_file:
        path_1 = ""

    if vectorizer_file in os.listdir("./upload"):
        path_2 = "./upload/" + vectorizer_file
        # vectorizer_file.save(path_2)
    else:
        return "No file vectorizer_file"

    if classifier_file in os.listdir("./upload"):

        path_3 = "./upload/" + classifier_file
        # classifier_file.save(path_3)
    else:
        return "No file classifier_file"

    if len(top_x_attributes) == 0 or top_x_attributes.isnumeric() != True:
        result = {"top_x_attributes": "should  not be equal to zero"}
        return jsonify(result, {"Status": 400})

    result = api.generate_entity_specific_positive_negative_attributes(path_1, int(top_x_attributes), Bigram_Boolean,
                                                                       path_2, path_3, data_json,list_of_entity_name)
    return jsonify(result)

@app.route('/auto_test_hidden_entity', methods=['GET', 'POST'])
def auto_test_hidden_entity():
    data = request.form
    files = request.files
    list_of_entity_name = data['list_of_entity_name']
    data_file = files['data_file']
    if data_file:
        path_1 = "./upload/" + data_file.filename
        data_file.save(path_1)
        if (path_1.endswith('.xlx') or path_1.endswith('.xlsx')):
            # print(file)
            df_entity_specific = pd.read_excel(path_1)
            df_entity_specific['Comment'] = df_entity_specific['Comment'].astype(str)
            df_entity_specific["Name"] = ""
            df_entity_specific["Result"] = ""
            df_entity_specific["Missing"] = ""
            result = api.searchHiddenEntity(df_entity_specific, list_of_entity_name)
            new_result = []
            for n in result:
                str_json = n.to_json()
                tmp_json = json.loads(str_json)
                tmp_json.pop('Name', None)
                new_result.append(tmp_json)
    return jsonify(new_result)

# ! logging the request json
@app.after_request
def after_request(response):
    """ Logging after every request. """
    # This avoids the duplication of registry in the log,
    # since that 500 is already logged via @app.errorhandler.
    if response.status_code != 500:
        ts = strftime('[%Y-%b-%d %H:%M]')
        logger.error('%s %s %s %s %s %s',
                      ts,
                      request.remote_addr,
                      request.method,
                      request.scheme,
                      request.full_path,
                      response.status
                      )
    return response
# ! logging  the internal error
@app.errorhandler(Exception)
def exceptions(e):
    """ Logging after every Exception. """
    ts = strftime('[%Y-%b-%d %H:%M]')
    tb = traceback.format_exc()
    logger.error('%s %s %s %s %s 5xx INTERNAL SERVER ERROR\nRequest json: %s \n%s',
                  ts,
                  request.remote_addr,
                  request.method,
                  request.scheme,
                  request.full_path,
                  request.get_json(),
                  tb)
    return "No Data Found!!!", 500

@app.route('/find_entity', methods=['GET', 'POST'])
def findEntity():
    data = request.form
    DEFAULT_ENTITY_CONST = ["practice","service", "exam room", "PCP", "everyone", "person", "practice manager", "manager", "location", "technicians", "nurses", "facility", "assistant", "desk", "rep", "lpn", "office", "rn", "establishment", "provider", "specialists", "employee", "nurse", "np", "receptionist", "pa", "employees", "staff", "attendant", "practitioner", "reception", "technician", "offices", "facilities", "assistants", "providers", "specialist", "staffs", "attendants", "practitioners", "lab", "front", "desk","wait time", "bedside manner", "telemedicine", "teleconference", "teledoc", "no wait"]
    list_of_entity_name = DEFAULT_ENTITY_CONST if ('list_of_entity_name' not in data) else data['list_of_entity_name']
    returnData = api.findEntity(list_of_entity_name)
    return jsonify(returnData, {"Status": 200})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)






