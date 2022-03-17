import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from dataset import get_news_dataset

class TfIdf_PCA():
    def __init__(self, pca_ndim=2):
        self.pca_ndim = pca_ndim
        self.tf_idf_vectorizer = TfidfVectorizer()
        self.PCA = PCA(n_components=self.pca_ndim)
    
    def fit_transform(self, X):
        tf_idf_vectors = self.tf_idf_vectorizer.fit_transform(X)
        tf_idf_vectors = tf_idf_vectors.toarray()
        pca_vectors = self.PCA.fit_transform(tf_idf_vectors)
        return pca_vectors
    
    def pca_eigenvector(self):
        return self.PCA.components_
    
    def predict(self, X):
        tf_idf_vectors = self.tf_idf_vectorizer.transform(X)
        tf_idf_vectors = tf_idf_vectors.toarray()
        pca_vectors = self.PCA.transform(tf_idf_vectors)
        return pca_vectors
    
    def fit(self, X):
        self.fit_transform(X)

def plot_news(pca_result, label):
    fake_result = pca_result[label == 0]
    real_result = pca_result[label == 1]
    plt.figure()
    plt.title('Fake News vs. Real News analysis using PCA')
    plt.scatter(fake_result[:, 0], fake_result[:, 1], c='r', label='fake news')
    plt.scatter(real_result[:, 0], real_result[:, 1], c='g', label='real news')
    plt.legend()

if __name__ == "__main__":
    fake_news, real_news = get_news_dataset()

    # Change Data Limit to 0 if we're going to go full model
    data_limit = 500
    # Limit the data (for low RAM capacity purpose)
    if data_limit:
        fake_news = fake_news[:data_limit]
        real_news = real_news[:data_limit]

    documents = np.concatenate((fake_news, real_news))
    labels = np.concatenate((np.zeros(len(fake_news)), np.ones(len(real_news))))

    model = TfIdf_PCA()
    result = model.fit_transform(documents)

    plot_news(result, labels)
    plt.show()