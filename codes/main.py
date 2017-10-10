from pandas import read_csv
from sklearn.externals import joblib
from sklearn.externals.joblib import Parallel, parallel_backend, register_parallel_backend
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

def importData():
    tweet_data = read_csv(filepath_or_buffer ="C:/Users/Ramanuja/Desktop/data.csv",header=None,skiprows=2,usecols = [1,2])# since no header info
#    filter_data = []
#    i = 0
#    for text in tweet_data[9]:
#        status(i,len(tweet_data[9]))
#        i = i + 1
#        filter_data.append(process_tweets(text))
#    y = tweet_data[10]
#    #vector = bagofWords(filter_data,1)
#    print "File Read Operation Completed"
    return tweet_data


data = importData()
vectorizer = CountVectorizer(decode_error='ignore',lowercase=True,
                             analyzer='word',stop_words='english',
                             max_df=1,min_df = 1 )
#parallel = Parallel(n_jobs=-1)
register_parallel_backend('parallel', Parallel)

corpus = data[1]
Y = data[2]
X, Y = shuffle(corpus, Y, random_state=0)
X = vectorizer.fit_transform(X)
ratio = .6
# training data
num_examples =len(Y)
X_train = X[1:int(num_examples*ratio)]
y_train = Y[1:int(num_examples*ratio)]

#test data
X_test = X[int(num_examples*ratio):]
y_test = Y[int(num_examples*ratio):]
clf = MLPClassifier(hidden_layer_sizes=(1000,500,100),activation='tanh',
                    solver='sgd', alpha=.01,tol = 10**-12,learning_rate='constant',
                    shuffle=False,max_iter=10000, verbose=True,random_state=1,
                    momentum =.9)
clf.fit(X_train, y_train)
#joblib.dump(clf, 'C:/Users/Ramanuja/Documents/python/Traffic Analysis/classifier/neural_net.pkl')
#clf = joblib.load('C:/Users/Ramanuja/Documents/python/Traffic Analysis/classifier/neural_net.pkl') 
data_x= vectorizer.transform(['#Bombay #HighCourt wants #traffic order during morchas https://t.co/SIN4xrsvuT']).toarray()

y_pred = []
#for i in X_test:    
#    temp = clf.predict(vectorizer.transform(i).toarray())
#    y_pred.append(temp[0])
