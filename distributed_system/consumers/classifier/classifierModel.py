import pickle
import numpy as np
import re,os
import inspect

class TweetClassifier():
    def __init__(self):
        # here is nice thread for the path : 
        #https://stackoverflow.com/questions/50499/how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executing/6628348#6628348
        self.file_name = 'classifier_data'
        self.current_path = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
        self.file = self.current_path+ "/" +self.file_name
        self.data = pickle.load(open(self.file, "rb"))
    def process_tweets(self,tweet):
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
        tweet = tweet.lower()
        tweet = re.sub('[\s]+', ' ', tweet)
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        tweet = pattern.sub(r"\1\1", tweet)
        tweet = re.sub('[^A-Za-z]+', ' ', tweet)
        tweet = tweet.strip('\'"')
        return re.split('\\W+',tweet)
    def makeFeatureVec(self,words, model):
        num_features = model.vector_size
        featureVec = np.zeros((num_features,),dtype="float32")
        nwords = 1.0
        index2word_set = set(model.wv.index2word)
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec,model[word])
        featureVec = np.divide(featureVec,nwords)
        return featureVec
    def getAvgFeatureVecs(self,tweets, model):
        counter = 0
        num_features = model.vector_size
        featureVecs = np.zeros((len(tweets),num_features),dtype="float32")
        for tweet in tweets:
           featureVecs[counter] = self.makeFeatureVec(tweet, model)
           counter = counter + 1
        return featureVecs
    def classify(self,tweet):
        classifier = self.data["classifier"]
        vec_model = self.data["vectr_model"]
        process_data = self.process_tweets(tweet)
        test_vec = self.getAvgFeatureVecs([process_data], vec_model)
        result = classifier.predict(test_vec)
        return result
if __name__ == '__main__':
    new_text = "This is for the test"
    obj = TweetClassifier()
    result = obj.classify(new_text)
    
    print(result[0])