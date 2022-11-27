import copy
import requests
import time
import json
import sys
start_time = time.time()


global_pos_list = ["able", "absolute", "absolutely", "abundant", "accessible", "accurate", "adept", "adequate",
                   "adorable",
                   "advanced", "affable", "affirmative", "alive", "amazing", "amenable", "amusing", "appealing",
                   "appreciated", "appreciative", "appropriate", "apt", "astonishing", "astounding", "astute",
                   "attendant",
                   "attentive", "attractive", "authentic", "autonomous", "available", "aware", "awesome",
                   "beautiful",
                   "best", "better", "bright", "brilliant", "calm", "candid", "capable", "certain",
                   "charismatic",
                   "charming", "cheerful", "cheery", "classic", "classy", "clean", "clear", "clever",
                   "coherent",
                   "comfortable", "comic", "competent", "complete", "complimentary", "comprehensible",
                   "concise",
                   "concrete", "confident", "considerable", "consistent", "consummate", "contemporary",
                   "convincing",
                   "cool", "courteous", "creative", "credible", "cultural", "cute", "decent", "deft",
                   "delightful",
                   "deserving", "detailed", "direct", "early", "easy", "economic", "educational", "effective",
                   "endearing",
                   "energetic", "engaging", "enjoyable", "enlightening", "entertaining", "enthusiastic", "epic",
                   "ethical",
                   "exact", "excellent", "exceptional", "excited", "exciting", "experienced", "exquisite",
                   "extraordinary",
                   "Efficient", "Empathetic", "efficient", "empathetic", "fabulous", "fair", "familiar",
                   "fantastic",
                   "fast", "favorite", "fine", "fit", "fitting", "flawless", "fresh", "friendly", "fun",
                   "funny",
                   "gentle",
                   "genuine", "gifted", "glad", "golden", "good", "grand", "great", "greatest", "handsome",
                   "happy",
                   "healthy", "hilarious", "honest", "humorous", "ideal", "impeccable", "important",
                   "impressive",
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
                   "never disrespectful", "never dull", "nice", "notable", "never efficient",
                   "never embarrased",
                   "never excessive", "never exemplary", "never expensive", "never female", "never frightened",
                   "never hurried", "never judged", "never judgmental", "never late", "never less",
                   "never long",
                   "never male", "never memorial", "never needless", "never negative", "never other",
                   "never overcrowded",
                   "never poor", "never pushy", "never random", "never rude", "never rush", "never rushed",
                   "never rushed!she", "never rusher", "never same", "never secondary", "never serious",
                   "never sick",
                   "never stupid", "never such", "never thorough", "never unanswered", "never uncomfortable",
                   "never unnecessary", "never urgent", "never worried", "never wrong", "new", "new.i",
                   "newborn",
                   "newer",
                   "newest", "next", "nice", "niceband", "nicedoctor", "nicer", "nicest", "nicotine", "nitrous",
                   "no",
                   "no asinine", "no bad", "no big", "no cervical", "no frustrating", "no insignificant",
                   "no little",
                   "no long", "no longer", "no medical", "no minor", "no necessary", "no next", "no sick",
                   "no silly",
                   "no small", "no stupid", "no triple", "no trivial", "no uninformed", "no worried",
                   "noligable",
                   "non",
                   "nonexistent", "nonjudgemental", "nonjudgmental", "nonreplaceable", "nonsense",
                   "nontraditional",
                   "normal", "not -", "not 1st", "not abrasive", "not acceptable", "not activate", "not active",
                   "not actual", "not additional", "not adverse", "not afraid", "not aggressive",
                   "not alarmist",
                   "not allergic", "not alone", "not alternate", "not amercian", "not ample", "not annual",
                   "not anti",
                   "not antibiotic", "not anxious", "not arbitrary", "not arrogant", "not autoimmune",
                   "not average",
                   "not awkward", "not back", "not bad", "not basic", "not bedside", "not big", "not brisk",
                   "not caring",
                   "not catastrophic", "not clinical", "not cold", "not common", "not compassionate",
                   "not competant",
                   "not complex", "not comprehensive", "not concerned", "not constructive", "not contagious",
                   "not convenient", "not convinced", "not correct", "not crazy", "not critical",
                   "not cumbersome",
                   "not current", "not curt", "not dandruff", "not dangerous", "not desirable", "not diabetic",
                   "not different", "not difficult", "not dire", "not dirty", "not disappointed",
                   "not dismissive",
                   "not dissatisfied", "not drs", "not due", "not eager", "not efficient", "not ego",
                   "not egotistical",
                   "not embarrassed", "not empathetic", "not emphatic", "not enough", "not entire",
                   "not expensive",
                   "not extra", "not fake", "not false", "not fare", "not female", "not few", "not fluid",
                   "not flush",
                   "not focused", "not forceful", "not further", "not generic", "not geriatric", "not glowing",
                   "not grateful", "not grueling", "not happier", "not hard", "not helpful", "not hesitant",
                   "not highest",
                   "not hippocratic", "not holistic", "not hurried", "not ibuprofen", "not ignorant",
                   "not immediate",
                   "not impatient", "not inclined", "not informative", "not informed", "not intermittent",
                   "not interpersonal", "not intimidated", "not intimidating", "not intrusive", "not involved",
                   "not judged", "not judgemental", "not judgmental", "not keen", "not knowledgeable",
                   "not last",
                   "not layman", "not least", "not less", "not likely", "not listened", "not little",
                   "not long",
                   "not longer", "not mean", "not medical", "not memmorial", "not memorial", "not minimal",
                   "not minor",
                   "not nagging", "not nameless", "not necesary", "not necessary", "not negative",
                   "not nervous",
                   "not neurological", "not next", "not nicer", "not nutrician", "not occupied", "not only",
                   "not open",
                   "not optimal", "not organized", "not other", "not ovarian", "not overbearing", "not overdue",
                   "not overwhelmed", "not painful", "not past", "not patient", "not patronizing",
                   "not pediatric",
                   "not personable", "not personal", "not personalible", "not physical", "not pink",
                   "not polite",
                   "not possible", "not pre", "not precsribe", "not prepared", "not prescribed", "not present",
                   "not pressing", "not pressured", "not pretentious", "not preventive", "not previous",
                   "not prior",
                   "not private", "not proactive", "not prompt", "not psychiatric", "not puahy", "not public",
                   "not pushy",
                   "not qualified", "not reactive", "not reassuring", "not receptive", "not regular",
                   "not related",
                   "not reluctant", "not reticent", "not rheumatoid", "not rude", "not rush", "not rushed",
                   "not same",
                   "not senior", "not serious", "not severe", "not short", "not sick", "not sicker",
                   "not sickness",
                   "not side", "not single", "not slow", "not softer", "not somber", "not sorry",
                   "not specific",
                   "not standard", "not sterile", "not stiff", "not stressful", "not stupid", "not such",
                   "not sufficient",
                   "not sugarcoat", "not supplant", "not technical", "not tense", "not terrible",
                   "not thankful",
                   "not third", "not thorough", "not time.negative", "not tolerable", "not touchy",
                   "not typical",
                   "not ugly", "not ultimate-", "not uncomfortable", "not undivided", "not unhappy",
                   "not unnecessary",
                   "not unprofessional", "not unrealistic", "not unsatisfactory", "not unsatisfied",
                   "not urgent",
                   "not usual", "not valid", "not verbal", "not wart", "not weekly", "not weird", "not well",
                   "not wince",
                   "not worried", "not worse", "not wrong", "noteworthy", "original", "outstanding", "peaceful",
                   "perfect",
                   "phenomenal", "pleasant", "pleased", "popular", "positive", "powerful", "precious",
                   "precise",
                   "pretty",
                   "Personable", "personable", "priceless", "professional", "profound", "pure", "quick",
                   "ready",
                   "realistic", "reasonable", "refreshing", "relevant", "remarkable", "resourceful",
                   "respectable",
                   "respectful", "responsible", "right", "safe", "satisfying", "seamless", "seasoned", "secure",
                   "sensitive", "sexy", "significant", "sincere", "skilled", "smart", "smile", "smooth",
                   "social",
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
                   "arbitrary", "atrocious", "average", "arrogant", "awful", "awkward", "bad", "bored",
                   "boring",
                   "broken", "casual",
                   "cluelessness", "cold", "complicated", "confused", "confusing", "corrupt", "crazy", "cruel",
                   "dangerous", "dark", "desperate", "difficult", "dirty", "disappointed", "disappointing",
                   "disgusting",
                   "dishonest", "distant", "disturbing", "dramatic", "dreadful", "dull", "dumb", "empty",
                   "erroneous",
                   "evil", "excessive", "expensive", "fake", "false", "fearful", "filthy", "flat", "forgetful",
                   "frightening", "frustrated", "frustrating", "green", "harsh", "horrible", "ill", "illegal",
                   "impatient",
                   "impossible", "incompetent", "inexperienced", "insecure", "insulting", "irrelevant", "late",
                   "lazy",
                   "limited", "lousy", "mad", "mean", "mediocre", "messy", "miserable", "narrow", "needless",
                   "negative",
                   "never able", "never better", "never certain", "never comfortable", "never direct",
                   "never easy",
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
                   "not excellent", "not exceptional", "not fair", "not familiar", "not fantastic",
                   "not favorite",
                   "not first", "not friendly", "not full", "not fun", "not general", "not gentle",
                   "not genuine",
                   "not good", "not great", "not greatest", "not happy", "not high", "not honest", "not hot",
                   "not important", "not impressed", "not intelligent", "not interested", "not kind",
                   "not loud",
                   "not major", "not many", "not more", "not most", "not much", "not natural", "not new",
                   "not nice",
                   "not normal", "not old", "not original", "not outstanding", "not own", "not particular",
                   "not perfect",
                   "not pleasant", "not pleased", "not positive", "not primary", "not professional",
                   "not proud",
                   "not quick", "not rare", "not ready", "not real", "not remarkable", "not respectful",
                   "not right",
                   "not satisfied", "not satisfying", "not secure", "not sensitive", "not sincere",
                   "not social",
                   "not strong", "not successful", "not sure", "not surprised", "not surprising",
                   "not sympathetic",
                   "not thrilled", "not true", "not useful", "not warm", "not whole", "not willing",
                   "not wonderful",
                   "not worth", "not worthy", "not young", "ordinary", "outraged", "painful", "pathetic",
                   "plain",
                   "pointless", "poor", "pretentious", "questionable", "repellent", "repetitive", "ridiculous",
                   "robotic",
                   "rough", "rude", "sad", "scary", "selfish", "shady", "shaky", "shallow", "shocked",
                   "shocking",
                   "shy",
                   "sick", "sickly", "silly", "simplistic", "sleepy", "sloppy", "slow", "sour", "stiff",
                   "strange",
                   "stupid", "tedious", "tense", "terrible", "tired", "tough", "ugly", "unanswered",
                   "unbelievable",
                   "uncomfortable", "unexplained", "unfortunate", "unhappy", "unhealthy", "unimportant",
                   "unknown",
                   "unlikely", "unnecessary", "unpleasant", "unpredictable", "unprofessional", "unrealistic",
                   "useless",
                   "usual", "vague", "weak", "weird", "worse", "worst", "wrong"]
list_of_entity_name = ["technicians", "nurses", "facility", "assistant", "desk", "rep", "lpn", "office", "rn",
                       "establishment", "provider", "specialists", "employee", "nurse", "np", "receptionist", "pa",
                       "employees", "staff", "attendant", "practitioner", "reception", "technician", "offices",
                       "facilities", "assistants", "providers", "specialist", "staffs", "attendants", "practitioners",
                       "lab", "practice manager", "manager"]
global_neg_list = str(global_neg_list)
global_pos_list = str(global_pos_list)
list_of_entity_name = str(list_of_entity_name)

def dataset(file):
    with open(file, encoding="utf8") as f:
        data = json.load(f)
        #data = f.readlines()
    print(type(data))
    return data

def batch_creation(content,n_o_b):
    #n_o_b ---> number of reviews in a batch

    file = []
    ch_content = copy.copy(content)
    #ch_content = content
    j = 1
    try:
        for i in range(len(ch_content)):
                temp = []
                for i in range(n_o_b):
                    temp.append(content[i])
                    ch_content.pop(i)
                    filename = "batch"+str(j)
                file.append(filename)
                with open('./batch/'+filename+'.json', 'w') as outfile:
                    data = temp
                    #data = '\n'.join([url, time_taken, data])
                    json.dump(data, outfile)
                j = j+1
                print("Original:{} Changed:{} TEmp:{}".format(len(content), len(ch_content), len(temp)))
    except:
        with open('./batch/' + filename + '.json', 'w') as outfile:
            data = ch_content
            # data = '\n'.join([url, time_taken, data])
            json.dump(data, outfile)
            #print(filename)
            file.append(filename)

        print("Original:{} Changed:{} TEmp:{}".format(len(content),len(ch_content),len(temp)))
    return file

def hitting_batch(filename,start_time):
    result={}
    with open('./batch/'+filename+'.json') as f:
        data = f.readlines()
        data = ''.join(data)
    #print (data)
    ploads = {'data_file': {}, 'data_json': data, "atr_thres_pos": "0", "atr_thres_neg": "0.1",
              "global_pos_list": global_pos_list, "global_neg_list": global_neg_list,
              "list_of_entity_name": list_of_entity_name
        , "vectorizer_filename": "vectorizer2020-02-03 06_02_10.629805.pkl",
              "classifier_filename": "classifier_model2020-02-03 06_02_10.629805.pkl"}
    r = requests.post('http://52.165.187.62:5000/extracthidden_score_generate_attributes', data=ploads)
    try:
        out = r.json()

    except:
        print("Error")
        out = r.text
    url = "URL: " + r.url

    t = time.time() - start_time
    tim = round(t, 2)
    time_taken = "time elapsed: " + str(tim) + "s"
    if r.status_code == 200:
        with open('./output/'+filename+'output.json', 'w') as outfile:
            data_out = out
            #data_out = '\n'.join([url, time_taken, data_out])
            #print(data_out)
            json.dump(data_out, outfile)
    result["filename"] = filename
    result["status_code"]=r.status_code
    result["time_taken"]=tim
    return result

if __name__ == "__main__":
    result =[]
    file = sys.argv[1]
    print("path:"+file)
    #Enter the number of item need to be in a single batch
    n_o_t = sys.argv[2]
    print("No. of item in single batch:"+str(n_o_t))
    #This below function would create the batch input file under batch folder
    #"C:/Users/Asus/Documents/repugen/backup/2.7.2020/f10k.txt"
    #"C:/Users/Asus/Documents/repugen/backup/2.7.2020/f10k.json"
    content = dataset(file)
    #print(type(content))
    print(len(content))
    #print(content)
    n_o_t = int(n_o_t)
    filename = batch_creation(content,n_o_t)
    #print(filename)
    for i in range(len(filename)):
        start_time = time.time()
        data = hitting_batch(filename[i],start_time)
        print(data)
        result.append(data)
    print(result)
    sum = 0
    for i in range(len(result)):

        timeee = result[i]['time_taken']
        timeee = float(timeee)
        sum = sum+timeee
    avg_time = sum/len(result)
    avg_time =round(avg_time,2)
    print("Average timetaken:{}s".format(avg_time))






