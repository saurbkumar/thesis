import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,NuSVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import LinearSVC
import re
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
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
    #plt.savefig(title, dpi = 100)
    plt.show()
    
filee=read_csv('F:/finalThreeCategories.csv',header=None,delimiter=',', usecols=[0,1])
data_= filee.get_values()

def process_tweets(tweet):
    #Convert www.* or https?://* to space
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    
    #Convert @username to username
   # tweet = re.sub('@([^\s]+)', r'\1',tweet)
    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub('#([^\s]+)', r'\1', tweet)
    
    #look for 2 or more repetitions of character and replace with the character itself
    #pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    #tweet = pattern.sub(r"\1\1", tweet)
    
    #Remove ... , -, special character
    tweet = re.sub('[^A-Za-z]+', ' ', tweet)
    #trim
    #tweet = tweet.strip('\'"')
    return tweet  

X = []
y = []
for i in data_:
    process_data = process_tweets(i[0])
    if len(process_data)>3:
        X.append(process_data)
        y.append(i[1])
        
vector = CountVectorizer()      
X_bag_of_words=vector.fit_transform(X)


title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=50, test_size=0.4, random_state=0)

title = "LogisticRegression"
estimator = LogisticRegression()
plot_learning_curve(estimator, title, X_bag_of_words, y,n_jobs=-1)

title = "LinearSVC"
estimator = LinearSVC()
plot_learning_curve(estimator, title, X_bag_of_words, y,n_jobs=-1)

title = "Non-Linear SVM:linear Kernel"
estimator = NuSVC(kernel='linear')
plot_learning_curve(estimator, title, X_bag_of_words, y,n_jobs=-1)

title = "Non-Linear SVM:poly kernel"
estimator = NuSVC(kernel='poly')
plot_learning_curve(estimator, title, X_bag_of_words, y,n_jobs=-1)

title = "Non-Linear SVM_poly kernel_degree:1"
estimator = NuSVC(kernel='poly', degree=1)
plot_learning_curve(estimator, title, X_bag_of_words, y,n_jobs=-1)

title = "Non-Linear SVM_poly kernel_degree:2"
estimator = NuSVC(kernel='poly', degree=2)
plot_learning_curve(estimator, title, X_bag_of_words, y,n_jobs=-1)

title = "Non-Linear SVM_poly kernel_degree:3"
estimator = NuSVC(kernel='poly', degree=3)
plot_learning_curve(estimator, title, X_bag_of_words, y,n_jobs=-1)

title = "Non-Linear SVM_poly kernel_degree:4"
estimator = NuSVC(kernel='poly', degree=4)
plot_learning_curve(estimator, title, X_bag_of_words, y,n_jobs=-1)

title = "Non-Linear SVM_poly kernel_degree:5"
estimator = NuSVC(kernel='poly', degree=5)
plot_learning_curve(estimator, title, X_bag_of_words, y,n_jobs=-1)

title = "Non-Linear SVM_rbf Kernel"
estimator = NuSVC(kernel='rbf')
plot_learning_curve(estimator, title, X_bag_of_words, y,n_jobs=-1)

title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = LinearSVC()
plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=1)

