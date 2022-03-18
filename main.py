import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from models import TextModel, k_fold_cross_validation
from dataset import get_news_dataset

pca_dim = 10
RandomForestTextPCA = TextModel(RandomForestClassifier(), pca_dim)
GradientBoostingTextPCA = TextModel(GradientBoostingClassifier(), pca_dim)
LogisticTextPCA = TextModel(LogisticRegression(), pca_dim)
RandomForestText = TextModel(RandomForestClassifier())
GradientBoostingText = TextModel(GradientBoostingClassifier())
LogisticText = TextModel(LogisticRegression())

fake_news, real_news = get_news_dataset()

# Change Data Limit to 0 if we're going to go full model
data_limit = 1000
# Limit the data (for low RAM capacity purpose)
if data_limit:
    fake_news = fake_news[:data_limit]
    real_news = real_news[:data_limit]

documents = np.concatenate((fake_news, real_news))
labels = np.concatenate((np.zeros(len(fake_news)), np.ones(len(real_news))))

k_fold = 5
train_dataset, test_dataset = k_fold_cross_validation(documents, k_fold)

def fit_model(documents, labels):
    RandomForestTextPCA.fit(documents, labels)
    GradientBoostingTextPCA.fit(documents, labels)
    LogisticTextPCA.fit(documents, labels)
    RandomForestText.fit(documents, labels)
    GradientBoostingText.fit(documents, labels)
    LogisticText.fit(documents, labels)

def evaluate_model(documents, labels):
    randomforestpca_acc = RandomForestTextPCA.evaluate(documents, labels)['accuracy']
    gradboostpca_acc = GradientBoostingTextPCA.evaluate(documents, labels)['accuracy']
    logisticpca_acc = LogisticTextPCA.evaluate(documents, labels)['accuracy']
    randomforest_acc = RandomForestText.evaluate(documents, labels)['accuracy']
    gradboost_acc = GradientBoostingText.evaluate(documents, labels)['accuracy']
    logistic_acc = LogisticText.evaluate(documents, labels)['accuracy']
    return {
        'randomforestpca': randomforestpca_acc,
        'gradboostpca': gradboostpca_acc,
        'logisticpca': logisticpca_acc,
        'randomforest': randomforest_acc,
        'gradboost': gradboost_acc,
        'logistic': logistic_acc
    }

for train, test in zip(train_dataset, test_dataset):
    xtrain, xtest, ytrain, ytest = documents[train], documents[test], labels[train], labels[test]
    fit_model(xtrain, ytrain)
    print(evaluate_model(xtest, ytest))