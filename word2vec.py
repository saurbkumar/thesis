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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap,locally_linear_embedding
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
from sklearn.cluster import DBSCAN
from textwrap import wrap


def wordPlot(x,y,tweet,data_lables):
    #tweet parameter - to mark the words on graph
    labels = ['{0}'.format(i) for i in tweet]
    s = [500 for n in range(len(x))]
    plt.subplots_adjust(bottom = 0.1)
    text = ''
    for item in tweet:
        text = text + item + " "
    plt.xlabel(text)
    plt.scatter(
        x, y, marker='o', c=x, s=s,
        cmap=plt.get_cmap('Spectral'))
    
    for label, x, y in zip(labels, x, y):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    
    plt.show()
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
    #tweet = re.sub('[\s]+', ' ', tweet)
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
    counter = 0.
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
       counter = counter + 1.
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
    tweet_data = read_csv(filepath_or_buffer ="data_old.csv",header=None,skiprows=2,usecols = [1,2])# since no header info
    tweet_text = tweet_data[1]
    tweet_tag = tweet_data[2]
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
    
    tweet_data = read_csv(filepath_or_buffer ="new_file.csv",header=None,skiprows=2,usecols = [3,4])# since no header info
    tweet_text = tweet_data[3]
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
num_features = 1000    # Word vector dimensionality                      
min_word_count = 5   # Minimum word count - window size                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-2   # Downsample setting for frequent words


model = word2vec.Word2Vec(X_train_word2vec, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
#model.init_sims(replace=True)
#for i in range(1000,6500,500):
#    num_feature = 100
model_name = "wordToVecModel_1000"# with 1000 num_feature
model.save(model_name)

# if want to load model 

model = word2vec.Word2Vec.load("wordToVecModel_1000")

# training and test data import

X, y = importTrainingData()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=42)

trainDataVecs = getAvgFeatureVecs( X_train, model, num_features )
#
testDataVecs = getAvgFeatureVecs( X_test, model, num_features )


def plotHistogram():
    plt.figure(figsize=(15,15))
    counter =1
    row = 5
    colm = 2
    plt.figure(figsize=(15,15))
    plt.subplots_adjust(hspace =.9)
    for i in range(1,row+1):
        for j in range(1,colm+1):
            ax = plt.subplot(row,colm,counter)
            plt.xlabel('Correlation Values')
            plt.ylabel('Frequency')    
            plt.hist(trainDataVecs[counter], facecolor='green',alpha = .75)
            title_text = ("\n".join(wrap('Histogram for ' + "\n"+ " ".join(X_train[counter]),50)))
            plt.title(title_text)
            plt.grid(True)
            mean = str(trainDataVecs[counter].mean())[:6]# 4 digit precision
            sigma = str(trainDataVecs[counter].std())[:6]# 4 digit precision
            str_mu = '$\mu='+mean+'$'
            str_sigma = '$\sigma='+sigma+'$'
            plt.text(.9,.8, str_mu,ha='center', va='center', transform=ax.transAxes,color='red', fontsize=12,fontweight=100)
            plt.text(.9,.65, str_sigma,ha='center', va='center', transform=ax.transAxes,color='red', fontsize=12)
            counter = counter + 1
    plt.savefig("Histogram with 1000 Features", dpi = 100)
    plt.show()
    
plotHistogram()

### Dimension Reduction using isomap

trans_data_train = Isomap(n_neighbors=10, n_components=2, eigen_solver='auto',tol=0,\
                      max_iter=None, path_method='auto', neighbors_algorithm='auto',\
                      n_jobs=-1).fit_transform(testDataVecs).T
################

#### PCA


trans_data_train1 = PCA(n_components=2).fit_transform(testDataVecs).T
#Training and test data plot

#train data plot 

#font = {'family': 'serif',
#        'color':  'darkred',
#        'weight': 'normal',
#        'size': 25,
#        }
#text = "Output for " + "Train"
#img_name = "Train"+".png"
#
#plt.figure(figsize=(15,15))
#plt.title(text, fontdict=font)
#color = 
#plt.scatter(trans_data_train[0], trans_data_train[1], c=color,alpha=0.5)
#plt.savefig(img_name, dpi = 400)
                      


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
def wordGraphPlot(model):
    pass


####for dimensionality reduction using isomap ###############
def plotGraph(data,y_pred,text,accuracy):
    
    #color = y_pred.T
    color = ["red"] * (len(y_pred))
    for i in xrange(len(y_pred)):
        if y_pred[i]==1:
            color[i]="green"
        else:
           color[i] = "red"
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 25,
        }
    text_ = "Output for " + text + " " + str(accuracy)
    img_name = text+".png"
    
    plt.figure(figsize=(15,15))
    plt.title(text_, fontdict=font)
    green_line = mlines.Line2D([], [], color='green', marker="o",
                          markersize=10, label='+ ve')
    red_line = mlines.Line2D([], [], color='red', marker="o",
                          markersize=10, label='- ve')
    plt.legend(handles=[green_line,red_line])
    plt.grid()
    plt.scatter(data[0], data[1],color = color,alpha=0.4)
    plt.savefig(img_name, dpi = 100)

  

clf_NN3 = MLPClassifier(solver='sgd',shuffle=False, activation = 'logistic',\
                    alpha=1e-5,hidden_layer_sizes=(60,6),\
                     random_state=1,momentum =1e-4,learning_rate ='adaptive',\
                     verbose =False,early_stopping =True)
y_pred_NN3,accuracy1 = clf_accuracy(clf_NN3)
text = "Neural_Network1"

clf_NN3 = MLPClassifier(solver='sgd',shuffle=False, activation = 'logistic',\
                    alpha=1e-5,hidden_layer_sizes=(600,60,6),\
                     random_state=1,momentum =1e-4,learning_rate ='adaptive',\
                     verbose =False,early_stopping =True)
y_pred_NN3,accuracy2 = clf_accuracy(clf_NN3)
text = "Neural_Network2"
print text+"Accuracy is", accuracy2 

clf_NN3 = MLPClassifier(solver='sgd',shuffle=False, activation = 'logistic',\
                    alpha=1e-5,hidden_layer_sizes=(1500,100,10),\
                     random_state=1,momentum =1e-4,learning_rate ='adaptive',\
                     verbose =False,early_stopping =True)
y_pred_NN3,accuracy3 = clf_accuracy(clf_NN3)
text = "Neural_Network3"  
print text+"Accuracy is", accuracy3
#plotGraph(trans_data_train,y_pred_NN3,text,accuracy1)
### Linear SVM #######################

clf_l_SVC = LinearSVC()
y_pred_l_SVC,accuracy4 = clf_accuracy(clf_l_SVC)
text = "Linear_SVM"
print text+"Accuracy is", accuracy4
#plotGraph(trans_data_train,y_pred_l_SVC,text,accuracy2)
### Non linear SVM
#kernal - gaussian #######################
#
clf_nl_SVC_gauss = NuSVC(tol  = 1e-4, random_state =100,kernel ='rbf')
y_pred_SVC_gauss,accuracy5 = clf_accuracy(clf_nl_SVC_gauss)
text = "SVC_gauss"
print text+"Accuracy is", accuracy5
#plotGraph(trans_data_train,y_pred_SVC_gauss,text,accuracy3)

#keran - poly 5 #######################
#degree - 5
clf_nl_SVC_ploy5 = NuSVC(tol  = 1e-4, random_state =100,kernel ='poly',degree= 5)
y_pred_SVC_ploy5,accuracy6 = clf_accuracy(clf_nl_SVC_ploy5)
text = "SVC_ploy5"
print text+"Accuracy is", accuracy6
#plotGraph(trans_data_train,y_pred_SVC_ploy5,text,accuracy4)
#degree - poly 3 #######################
clf_nl_SVC_ploy3 = NuSVC(tol  = 1e-4, random_state =100,kernel ='poly',degree= 3)
y_pred_SVC_ploy3,accuracy7 = clf_accuracy(clf_nl_SVC_ploy3)
text = "SVC_ploy3"
print text+"Accuracy is", accuracy7
#plotGraph(trans_data_train,y_pred_SVC_ploy3,text,accuracy5)
#degree - ploy 2 #######################
clf_nl_SVC_ploy2 = NuSVC(tol  = 1e-4, random_state =100,kernel ='poly',degree= 2)
y_pred_SVC_ploy2,accuracy8 = clf_accuracy(clf_nl_SVC_ploy2)
text = "SVC_ploy2"
print text+"Accuracy is", accuracy8
#plotGraph(trans_data_train,y_pred_SVC_ploy2,text,accuracy6)
#degree - ploy 1 #######################
clf_nl_SVC_ploy1 = NuSVC(tol  = 1e-4, random_state =100,kernel ='poly',degree= 1)
y_pred_SVC_ploy1,accuracy9 = clf_accuracy(clf_nl_SVC_ploy1)
text = "SVC_ploy1"
print text+"Accuracy is", accuracy9
#plotGraph(trans_data_train,y_pred_SVC_ploy1,text,accuracy7)
### logistic regression ######

clf_lr = LogisticRegression(max_iter =1000,random_state=10,tol=1e-4,n_jobs= -1)
y_pred_clf_lr,accuracy10 = clf_accuracy(clf_lr)
text = "clf_lr"
print text+"Accuracy is", accuracy10
#plotGraph(trans_data_train,y_pred_clf_lr,text,accuracy8)
#plt_confusion_matrix(y_pred_NN3,text,accuracy1)

