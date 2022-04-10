import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from wordcloud import WordCloud, STOPWORDS
from dataset import get_news_dataset

fake_news, real_news = get_news_dataset()
fake_news = "\n".join(fake_news)
real_news = "\n".join(real_news)

STOPWORDS = set(list(STOPWORDS) + ['Donald Trump', 'Donald', 'Trump', 'White House', 'Hillary Clinton'])

fake_cloud = WordCloud(stopwords=STOPWORDS).generate(fake_news)

real_cloud = WordCloud(stopwords=STOPWORDS).generate(real_news)

plt.title('Fake News WordCloud EDA')
plt.imshow(fake_cloud, interpolation='bilinear')
plt.imsave('fakenews_wordcloud.png', fake_cloud)
plt.figure()
plt.title('Real News WordCloud EDA')
plt.imshow(real_cloud, interpolation='bilinear')
plt.imsave('realnews_wordcloud.png', real_cloud)
plt.axis('off')

# PCA
class TextDimRed():
    def __init__(self, dim_red_type, limit_dim):
        self.Tfidf = TfidfVectorizer(max_df=0.8, min_df=0.2)
        self.dim_red_model = dim_red_type(n_components=limit_dim)
    
    def fit_transform(self, X, Y):
        text_tfidf = self.Tfidf.fit_transform(X)
        text_tfidf = text_tfidf.toarray()
        text_tfidf = self.dim_red_model.fit_transform(text_tfidf)
        return text_tfidf
    
    def transform(self, X):
        text_tfidf = self.Tfidf.transform(X)
        text_tfidf = text_tfidf.toarray()
        text_tfidf = self.dim_red_model.transform(text_tfidf)
        return text_tfidf

    def plot(self, X, Y):
        text_tfidf = self.transform(X)
        plt.scatter(text_tfidf[Y == 0][:, 0], text_tfidf[Y == 0][:, 1], c='r')
        plt.scatter(text_tfidf[Y == 1][:, 0], text_tfidf[Y == 1][:, 1], c='g')

pca_dim_red = TextDimRed(PCA, 2)
svd_dim_red = TextDimRed(TruncatedSVD, 2)

fake_news, real_news = get_news_dataset()
data_limit = 100
# Limit the data (for low RAM capacity purpose)
if data_limit:
    fake_news = fake_news[:data_limit]
    real_news = real_news[:data_limit]

documents = np.concatenate((fake_news, real_news))
labels = np.concatenate((np.zeros(len(fake_news)), np.ones(len(real_news))))

pca_dim_red.fit_transform(documents, labels)
svd_dim_red.fit_transform(documents, labels)
plt.figure()
plt.title('PCA')
pca_dim_red.plot(documents, labels)
plt.savefig('pca.png')
plt.figure()
plt.title('Trancated SVD')
svd_dim_red.plot(documents, labels)
plt.savefig('svd.png')
plt.show()