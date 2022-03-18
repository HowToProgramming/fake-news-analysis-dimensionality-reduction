import numpy as np
from numpy.random import choice
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

def k_fold_cross_validation(dataset, k=10):
    n = len(dataset)
    np_range = np.array(range(n))
    test_dataset = choice(np_range, size=(n // k, k), replace=False)
    train_dataset = []
    for test in test_dataset:
        train_dataset.append(np.delete(np_range, test))
    
    return np.array(train_dataset), test_dataset

class TextModel():
    def __init__(self, classifier_model, pca_dim=None):
        self.Tfidf = TfidfVectorizer()
        self.pca = PCA(n_components=pca_dim) if pca_dim else None
        self.classifier_model = classifier_model
    
    def fit(self, X, Y):
        text_tfidf = self.tf_idf.fit_transform(X)
        text_tfidf = text_tfidf.toarray()
        if self.pca:
            text_tfidf = self.pca.fit_transform(text_tfidf)
        self.classifier_model.fit(text_tfidf, Y)
    
    def predict(self, X):
        text_tfidf = self.tf_idf.transform(X)
        text_tfidf = text_tfidf.toarray()
        if self.pca:
            text_tfidf = self.pca.transform(text_tfidf)
        return self.classifier_model.predict(text_tfidf)
    
    def evaluate(self, X, Y):
        y_hat = self.predict(X)
        accuracy = len(np.where(y_hat == Y)[0]) / len(Y)
        
        return {'accuracy': accuracy}