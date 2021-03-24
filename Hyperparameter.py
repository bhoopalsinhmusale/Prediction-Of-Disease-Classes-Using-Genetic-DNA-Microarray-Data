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
results = pd.DataFrame(model.cv_results_