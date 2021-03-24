#####################################
# Group Members                     #
# Bhoopalsinh Musale	002269332   #
# Syed Malik Muzaffar	002269955   #
#####################################


# Imports
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import scipy.stats as st

from sklearn import tree
from sklearn import naive_bayes
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tempfile import mkdtemp
from TScoreSelection import TScoreSelection
import os
import warnings
warnings.filterwarnings("ignore")


def load_data():
    '''
        Load data from CSV file
        returns X,Y and random seeds
    '''
    dataFrame = pd.read_csv('pp5i_train.gr.csv')
    dataFrame.set_index('SNO', inplace=True)
    dataFrame = dataFrame.transpose()
    dataFrame.reset_index(drop=True, inplace=True)

    y = pd.read_csv('pp5i_train_class.txt')
    dataFrame = pd.concat([dataFrame, y], axis=1)
    myRndSeeds = 72
    dataFrame = dataFrame.sample(
        frac=1, random_state=myRndSeeds).reset_index(drop=True)
    print(dataFrame.shape)
    print(dataFrame.head())

    X = dataFrame.drop('Class', axis=1)

    y = dataFrame['Class']

    return X, y, myRndSeeds


def clean_data(X):
    '''
        Thresholding both train and test data 
        to a minimum value of 20, maximum of 16,000.
    '''
    X.clip(upper=16000, lower=20, inplace=True)
    print(X.shape)
    X = X.loc[:, X.max() - X.min() > 2]
    print(X.shape)
    return X


if __name__ == "__main__":
    # Loading Dataset
    X, y, myRndSeeds = load_data()

    # Cleaning Dataset
    X = clean_data(X)

    # Feature selection using Ttest
    cachedir = mkdtemp()
    pipe = Pipeline([('featureSelection', TScoreSelection(w=10)),
                     ('classify', KNeighborsClassifier(n_neighbors=1))],
                    memory=cachedir)

    # Top Gene Selection
    N_GENES = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
    N_LAYERS = [(32,), (64,), (128,)]

    # Hyperparameter Optimization
    tuned_parameters = [
        # KNN Classifier(2,3,4)
        {'featureSelection__w': N_GENES,
         'classify': [KNeighborsClassifier()],
         'classify__n_neighbors': [2, 3, 4]
         },
        # Decision Tree Classifier(J48 algorithm)
        {'featureSelection__w': N_GENES,
         'classify': [tree.DecisionTreeClassifier()],
         'classify__criterion':['gini', 'entropy'],
         'classify__min_samples_leaf': [1, 3, 5],
         'classify__max_depth': [3, 6, 9],
         'classify__presort': [True]
         },
        # Neural Network Multi-label Classifier
        {'featureSelection__w': N_GENES,
         'classify': [MLPClassifier()],
         'classify__hidden_layer_sizes': N_LAYERS,
         'classify__activation': ['logistic'],
         'classify__alpha':[0.05, 0.01, 0.005, 0.001],
         'classify__max_iter':[1000],
         'classify__solver': ['lbfgs'],
         'classify__verbose': [True]
         },
        # Naïve Bayes Classifier
        {'featureSelection__w': N_GENES,
         'classify': [naive_bayes.GaussianNB()]
         },
        # AdaBoost Classifier
        {'featureSelection__w': N_GENES,
         'classify': [AdaBoostClassifier()]
         }
    ]

    # Model Selection using Pipeline and Cross validation
    kfolds = KFold(n_splits=5, shuffle=True, random_state=myRndSeeds)
    model = GridSearchCV(pipe, tuned_parameters, cv=kfolds,
                         return_train_score=True)
    model.fit(X, y)
    results = pd.DataFrame(model.cv_results_)

    print(results.sort_values(by='mean_test_score', ascending=False).head())

    # Best Model
    best_estimator_ = model.best_estimator_
    print(best_estimator_)

    # Running best model on Test dataset
    testDataFrame = pd.read_csv('pp5i_test.gr.csv')
    testDataFrame.set_index('SNO', inplace=True)
    X_test = testDataFrame.transpose()
    X_test.reset_index(drop=True, inplace=True)

    # Generating output Y for given Test Dataset
    Y = pd.DataFrame()
    Y['predicted'] = model.predict(X_test)
    finalResult = Y

    # Final Output
    print(finalResult)
