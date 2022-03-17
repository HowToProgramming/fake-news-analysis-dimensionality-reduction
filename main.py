import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from dataset import get_news_dataset

def tfidf_pca(dataset, label, ndim=2):
    tfidf_text = TfidfVectorizer()
    result = tfidf_text.fit_transform(dataset)
    word_list = tfidf_text.vocabulary_
    word_inv = {v: k for k, v in word_list.items()}
    result = result.toarray()
    pca_text = PCA(n_components=ndim)
    pca_result = pca_text.fit_transform(result)
    fake_result = pca_result[np.where(label == 0)]
    real_result = pca_result[np.where(label == 1)]
    eigenvector = pca_text.components_
    maxes = np.argsort(eigenvector, axis=1)[:, :5]
    max_word = []
    for max_values in maxes:
        a = []
        for idx in max_values:
            a.append(word_inv[idx])
        max_word.append(a.copy())
    print(max_word)
    plt.figure()
    plt.title('Fake News vs. Real News analysis using PCA')
    plt.scatter(fake_result[:, 0], fake_result[:, 1], c='r', label='fake news')
    plt.scatter(real_result[:, 0], real_result[:, 1], c='g', label='real news')
    plt.legend()

fake_news, real_news = get_news_dataset()

data_limit = 1000
fake_news = fake_news[:data_limit]
real_news = real_news[:data_limit]

tfidf_pca(np.concatenate((fake_news, real_news)), np.concatenate((np.zeros(len(fake_news)), np.ones(len(real_news)))))
plt.savefig('fake_real.png')
plt.show()

# This model
# Compare with state of the art using k-fold cross validation