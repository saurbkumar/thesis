
import sys,re
import logging
import itertools
import numpy as np
from pandas import read_csv
from collections import deque,Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from gensim.models import word2vec
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap,locally_linear_embedding
from sklearn.decomposition import PCA
import matplotlib.lines as mlines
from sklearn.model_selection import learning_curve

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Credit : Sklearn - http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py    
    """
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
    #featureVec = np.divide(featureVec,nwords)
    return featureVec
def makeFeatureVecCluster_other(words, model, num_features):
    pass
def makeFeatureVecCluster_kmean(words, model, num_features):
    
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
    

    data_points = []
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            data_points.append(model[word])# finding out the word vector reperesentation from a model
            #featureVec = np.add(featureVec,model[word])
    #clustering
    if len(data_points)>1:
        kmeans = KMeans(n_clusters=2, random_state=0, n_jobs=-1).fit(data_points)
        class_labels = kmeans.labels_
        class_labels_count = Counter(class_labels)
        relevant_words = np.where(class_labels ==class_labels_count[max(class_labels_count.values())])# return the class having max point
        relevant_words = list(relevant_words[0])
        for j in relevant_words:
            featureVec = np.add(featureVec,data_points[j])
    else:
        for j in data_points:
            featureVec = np.add(featureVec,j)
    
        #data_.append(data_points[j])
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec
def getFeatureVecs(tweets, model, num_features):
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
    total_len = len(tweets)
    for tweet in tweets:
       
       # Call the function (defined above) that makes average feature vectors
       featureVecs[counter] = makeFeatureVec(tweet, model, \
           num_features)
       
       # Increment the counter
       counter = counter + 1.
       status(counter,total_len)
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
    total_len = len(tweet_text)
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
model.init_sims(replace=True)
model_name = "wordToVecModel"
model.save(model_name)

# if want to load model 

model = word2vec.Word2Vec.load("wordToVecModel_1000")

# training and test data import

X, y = importTrainingData()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=42)

trainDataVecs = getFeatureVecs( X_train, model, num_features )
#
testDataVecs = getFeatureVecs( X_test, model, num_features )

X_input = getFeatureVecs( X, model, num_features )

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
    
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

#    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                     train_scores_mean + train_scores_std, alpha=0.1,
#                     color="r")
#    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(title, dpi = 100)
    plt.show()


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
                        
clf_NN3 = MLPClassifier(solver='sgd',shuffle=False, activation = 'logistic',\
                    alpha=1e-5,hidden_layer_sizes=(1500,100,10),\
                     random_state=1,momentum =1e-4,learning_rate ='adaptive',\
                     verbose =False,early_stopping =True)
#y_pred_NN3,accuracy1 = clf_accuracy(clf_NN3)
text = "Neural_Network"
plot_learning_curve(clf_NN3, text, X_input, y,n_jobs=-1)
#plotGraph(trans_data_train,y_pred_NN3,text,accuracy1)
### Linear SVM #######################

clf_l_SVC = LinearSVC()
#y_pred_l_SVC,accuracy2 = clf_accuracy(clf_l_SVC)
text = "Linear_SVM"
plot_learning_curve(clf_l_SVC, text, X_input, y,n_jobs=-1)
#plotGraph(trans_data_train,y_pred_l_SVC,text,accuracy2)
### Non linear SVM
#kernal - gaussian #######################
#
clf_nl_SVC_gauss = NuSVC(tol  = 1e-4, random_state =100,kernel ='rbf')
#y_pred_SVC_gauss,accuracy3 = clf_accuracy(clf_nl_SVC_gauss)
text = "SVC_gauss"
plot_learning_curve(clf_nl_SVC_gauss, text, X_input, y,n_jobs=-1)
#plotGraph(trans_data_train,y_pred_SVC_gauss,text,accuracy3)
#keran - poly
#degree - 5
clf_nl_SVC_ploy5 = NuSVC(tol  = 1e-4, random_state =100,kernel ='poly',degree= 5)
#y_pred_SVC_ploy5,accuracy4 = clf_accuracy(clf_nl_SVC_ploy5)
text = "SVC_ploy5"
plot_learning_curve(clf_nl_SVC_ploy5, text, X_input, y,n_jobs=-1)
#plotGraph(trans_data_train,y_pred_SVC_ploy5,text,accuracy4)
#degree - 3
clf_nl_SVC_ploy3 = NuSVC(tol  = 1e-4, random_state =100,kernel ='poly',degree= 3)
#y_pred_SVC_ploy3,accuracy5 = clf_accuracy(clf_nl_SVC_ploy3)
text = "SVC_ploy3"
plot_learning_curve(clf_nl_SVC_ploy5, text, X_input, y,n_jobs=-1)
#plotGraph(trans_data_train,y_pred_SVC_ploy3,text,accuracy5)
#degree - 2
clf_nl_SVC_ploy2 = NuSVC(tol  = 1e-4, random_state =100,kernel ='poly',degree= 2)
#y_pred_SVC_ploy2,accuracy6 = clf_accuracy(clf_nl_SVC_ploy2)
text = "SVC_ploy2"
plot_learning_curve(clf_nl_SVC_ploy2, text, X_input, y,n_jobs=-1)
#plotGraph(trans_data_train,y_pred_SVC_ploy2,text,accuracy6)
#degree - 1
clf_nl_SVC_ploy1 = NuSVC(tol  = 1e-4, random_state =100,kernel ='poly',degree= 1)
#y_pred_SVC_ploy1,accuracy7 = clf_accuracy(clf_nl_SVC_ploy1)
text = "SVC_ploy1"
plot_learning_curve(clf_nl_SVC_ploy1, text, X_input, y,n_jobs=-1)
#plotGraph(trans_data_train,y_pred_SVC_ploy1,text,accuracy7)

### logistic regression ######

clf_lr = LogisticRegression(max_iter =1000,random_state=10,tol=1e-4,n_jobs= -1)
#y_pred_clf_lr,accuracy8 = clf_accuracy(clf_lr)
text = "clf_lr"
plot_learning_curve(clf_lr, text, X_input, y,n_jobs=-1)
#plotGraph(trans_data_train,y_pred_clf_lr,text,accuracy8)
#plt_confusion_matrix(y_pred_NN3,text,accuracy1)

