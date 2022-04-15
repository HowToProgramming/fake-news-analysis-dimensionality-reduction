# life
import joblib
import numpy as np

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from dataset import get_news_dataset


# The real question is what is the meaning of life
# am I real ? am I exist ?
# why am I still here just to suffer
# my life sucks and it is full of pain

# man what the fuck am I doing
# constantly finding the meaning of life ?
# or do something else
# whatever

# models

tfidf = TfidfVectorizer(min_df=0.2, max_df=0.8)
pca = PCA(n_components=10)
svd = TruncatedSVD(n_components=10)
randomforest_pca = RandomForestClassifier()
logisticreg_pca = LogisticRegression()
randomforest_svd = RandomForestClassifier()
logisticreg_svd = LogisticRegression()

# Load Dataset

fake_news, real_news = get_news_dataset()

data_limit = 10000
# Limit the data (for low RAM capacity purpose)
if data_limit:
    fake_news = fake_news[:data_limit]
    real_news = real_news[:data_limit]

documents = np.concatenate((fake_news, real_news))
labels = np.concatenate((np.zeros(len(fake_news)), np.ones(len(real_news))))

# yeah just fit the model nothing to see here

tfidfres = tfidf.fit_transform(documents)
pcares = pca.fit_transform(tfidfres.toarray())
svdres = svd.fit_transform(tfidfres)

randomforest_pca.fit(pcares, labels)
randomforest_svd.fit(svdres, labels)

logisticreg_pca.fit(pcares, labels)
logisticreg_svd.fit(svdres, labels)

model_fp = 'models/'

def save_model(model, model_name):
    fp = model_fp + model_name + ".sva"
    with open(fp, 'w+') as f:
        joblib.dump(model, fp)
        f.close()
    
save_model(tfidf, 'tfidf')
save_model(pca, 'pca')
save_model(svd, 'svd')
save_model(randomforest_pca, 'randomforest_pca')
save_model(randomforest_svd, 'randomforest_svd')
save_model(logisticreg_pca, 'logisticreg_pca')
save_model(logisticreg_svd, 'logisticreg_svd')