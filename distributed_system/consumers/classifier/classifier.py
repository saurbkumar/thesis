import sys,re
import logging
import itertools
import numpy as np
from pandas import read_csv
from collections import deque
import matplotlib.pyplot as plt
from gensim.models import word2vec
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap,locally_linear_embedding
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
from sklearn.cluster import DBSCAN
from textwrap import wrap
import pickle

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
   
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def process_tweets(tweet):
#    text =  text.decode('utf-8')
#    tknzr = TweetTokenizer()
    #Convert www.* or https?://* to space

    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    tweet = tweet.lower()
    #Convert @username to username
    #tweet = re.sub('@([^\s]+)', r'\1',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    #tweet = re.sub('#([^\s]+)', r'\1', tweet)
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    tweet = pattern.sub(r"\1\1", tweet)
    #Remove ... , -, special character
    tweet = re.sub('[^A-Za-z]+', ' ', tweet)
    #trim
    tweet = tweet.strip('\'"')
    
    return re.split('\\W+',tweet)
def makeFeatureVec(words, model, num_features):
    
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    
    #Min case
    nwords = 1.0
    
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec
def getAvgFeatureVecs(tweets, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0
    #
    # Preallocate a 2D numpy array, for speed
    featureVecs = np.zeros((len(tweets),num_features),dtype="float32")
    #
    # Loop through the reviews
    for tweet in tweets:
       
       # Call the function (defined above) that makes average feature vectors
       featureVecs[counter] = makeFeatureVec(tweet, model, \
           num_features)
       
       # Increment the counter
       counter = counter + 1
    return featureVecs
def status(i,num_passe):
    barLength = 20 
    status = ""
    progress = (float(i)/(num_passe-1))
    block = int(round(barLength*progress))
    sys.stdout.write('\r')
    text = "[{0}] File Read {1}% Completed.".format( "#"*block + "-"*(barLength-block), format(progress*100,".2f"),status)
    sys.stdout.write(text)
    sys.stdout.flush()
    
def importTrainingData():
    # to train the classifier
    '''
    read data and tokenized it.
    data = [@anth0nypears0n Hi Anthony, no they don't. Children under 11 years travel 
     for free when travelling with a fare paying adult."]
    output = ['@anth0nypears0n', 'Hi', 'Anthony', no', 'they',.....],[]...]
    '''
    tweet_data = read_csv(filepath_or_buffer ="D:/Document/python/Barco/barco/data/surely_data__my_data_combined.csv",header=None,skiprows=2,usecols = [0,1])# since no header info
    tweet_text = tweet_data[0]
    tweet_tag = tweet_data[1]
    filter_data = deque()
    i = 1
    for text in tweet_text:
        #status(i,len(tweet_text))
        filter_data.append(process_tweets(text))
        i = i + 1
    print "File Read Operation Completed"
    return filter_data,tweet_tag
def importAnnotateedData():
    # to import the annotate data for word to vec training
    
    tweet_data = read_csv(filepath_or_buffer ="D:/Document/python/Barco/barco/data/surely_data__my_data_combined.csv",header=None,skiprows=2,usecols = [0,1])# since no header info
    tweet_text = tweet_data[0]
    #tweet_tag = tweet_data[2]
    filter_data = deque()
    i = 1
    for text in tweet_text:
        filter_data.append(process_tweets(text))
        i = i + 1
        #status(i,total_len)
    return filter_data
    
# this is to train wor2vec model
X_train_word2vec = importAnnotateedData()


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
class_names  = ["No Traffic info in Tweets","Traffic Info in Tweets"]
num_features = 300    # Word vector dimensionality                      
min_word_count = 3   # Minimum word count - window size                        
num_workers = 4       # Number of threads to run in parallel
context = 3      # Context window size                                                                                    
downsampling = 1e-2   # Downsample setting for frequent words


model = word2vec.Word2Vec(X_train_word2vec, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
#model.init_sims(replace=True)
#for i in range(1000,6500,500):
#    num_feature = 100
model_name = "VecModel_Window_3_Feature_1000"# with 1000 num_feature
model.save(model_name)

# if want to load model 

#model = word2vec.Word2Vec.load("VecModel_Window_3_Feature_1000")
#
## training and test data import

X, y = importTrainingData()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=42)

trainDataVecs = getAvgFeatureVecs( X_train, model, num_features )
#
testDataVecs = getAvgFeatureVecs( X_test, model, num_features )



#
def plt_confusion_matrix(y_pred,text,accuracy):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test.get_values(), y_pred)
    img_name = "confusion_matrix" + text
    # Plot non-normalized confusion matrix
    title_= 'Confusion matrix, without normalization' + "Accuracy is "+ str(accuracy)
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title=title_)
    plt.savefig(img_name, dpi = 400)
## function to calculate the accuracy of classifier

def clf_accuracy(clf):
###############Accuracy score#################

    clf.fit(trainDataVecs, y_train.get_values())
    y_pred = clf.predict(testDataVecs)
    score_ = accuracy_score(y_test.get_values(), y_pred,normalize=True)
    #print "Accuracy is ", score_
    #plt_confusion_matrix(y_pred)
    return y_pred,score_


#kernal - gaussian #######################
#
clf_nl_SVC_gauss = NuSVC(tol  = 1e-4, random_state =100,kernel ='rbf')
clf_nl_SVC_gauss.fit(trainDataVecs, y_train.get_values())
y_pred = clf_nl_SVC_gauss.predict(testDataVecs)

data = {'classifier':clf_nl_SVC_gauss,'vectr_model':model}


# save a model only one time
f = open('classifier_data', 'wb')
pickle.dump(data, f)
f.close()


# #open file 
#f = open("classifier_data",'rb')
#data = pickle.load(f)
#f.close()

new_text = "This is for the test"
process_data = process_tweets(new_text)
classifier = data["classifier"]
vec_model = data["vectr_model"]
test_vec = getAvgFeatureVecs([process_data], vec_model)
result = classifier.predict(test_vec)