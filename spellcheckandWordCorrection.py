import re
from collections import Counter
from pandas import read_csv
from nltk.corpus import stopwords

####### Spell Correction #######

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('data/big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


####### Words Correctionns #######


test = "this Is @ for Ave. @ nasaasd"

def preProcess(string):
    '''
    Remove ., _@_ -> at, [A-Z] -> lower case
    '''
    string = list(string)
    for index in range(len(string)):
        '''
        Lower case conversion
        '''
        string[index] = string[index].lower()
        if string[index]=='@':
            '''
            Remove _@_ with at -> _ is spcae
            This is for better location detection
            '''
            if string[index-1] == " " and string[index+1] == " ":
                string[index]='at'
        if string[index]==".":
            """
            Remove "." at the end of the abbreviation like Ave.| Highw.
            This is for better abbreviation detection
            """
            string[index]=""
    string = "".join(string)
    return string
def data_read():
    '''
    Read acronym and abbreviation data
    '''
    acronym_data = read_csv(filepath_or_buffer ="data/acronym.csv",header=None,skiprows=0)# since no header info
    acrr = [acronym_data[0].tolist(),acronym_data[1].tolist(),acronym_data[2].tolist()]
    
    abbr_data = read_csv(filepath_or_buffer="data/Abbreviations.csv",header=None,skiprows=0)
    abbrr = [abbr_data[0].tolist(),abbr_data[1].tolist()]
    return acrr,abbrr

acronym_data,abbr_data = data_read()

acronym = acronym_data[0]
acronym_defination = acronym_data[1]

abbr = abbr_data[1]
abbr_defination = abbr_data[0]


def linear_search(word,abbr):
    found = False
    for index,abbrs in enumerate(abbr):
        if abbrs == word:
            found = True
            break
    if found:
        return index
    else:
        return None
stopWords = set(stopwords.words('english'))
temp = preProcess(test)
output_ = []
for word in temp.split():
     index = linear_search(word,abbr)
     if index is not None:
         '''
         Element found in abbreviations
         '''
         output_.append(abbr_defination[index])
     else:
         '''
         Element is not in abbreviation list
         if not a stop word then check if lenght is in 3-5 range,
         if it is check for the acronym, else append directly
         '''
         if word not in stopWords:
             if len(word) >=2 and len(word)<=5:
                 index = linear_search(word,acronym)
                 if index is not None:
                     output_.append(acronym_defination[index])
             else:
                 output_.append(word)
         else:
                 output_.append(word)
     
print(output_)  
    
    
    
    
    
    
    
    
    
    
    
    