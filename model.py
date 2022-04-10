import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score

MAX_DIM = 10

classification_model = {
        'GaussianNaiveBayes': GaussianNB,
        'RandomForest': RandomForestClassifier,
        'LogisticRegression': LogisticRegression,
        'GradientBoostClassifier': GradientBoostingClassifier
    }

def create_models():
    tfidf_model = [('tfidf', TfidfVectorizer(max_df=0.8, min_df=0.2))]
    dim_reduction_models = []
    classification_models = []
    for dim in range(2, MAX_DIM + 1):
        dim_reduction_models.append((f'pca_{dim}', PCA(n_components=dim)))
        dim_reduction_models.append((f'truncated_svd_{dim}', TruncatedSVD(n_components=dim)))

    for m_label, model in classification_model.items():
        classification_models.append((m_label, model()))

    pipes = {}
    to_dense = ('to_dense', FunctionTransformer(lambda x: x.todense(), accept_sparse=True))
    for tfidf in tfidf_model:
        
        for dim_red in dim_reduction_models + [None]:
            
            for classification in classification_models:
                tfidf_lbl = tfidf[0]
                dim_lbl = dim_red[0] if dim_red else ''
                classi_lbl = classification[0]
                steps = [tfidf, to_dense, classification]
                if dim_red:
                    steps = [tfidf, to_dense, dim_red, classification]
                model_lbl = "_".join([tfidf_lbl, dim_lbl, classi_lbl])
                pipes[model_lbl] = Pipeline(steps=steps)

    print(f"Created {len(pipes)} models")
    return pipes

def create_dim_red_visualize_models():
    tfidf_model = ('tfidf', TfidfVectorizer(max_df=0.8, min_df=0.2))
    pca = (f'pca', PCA(n_components=2))
    svd = (f'svd', TruncatedSVD(n_components=2))
    return Pipeline(steps=[tfidf_model, pca]), Pipeline(steps=[tfidf_model, svd])


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return n_scores