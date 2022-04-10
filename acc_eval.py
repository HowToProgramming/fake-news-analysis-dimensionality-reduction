import matplotlib.pyplot as plt
import json
import numpy as np

from model import classification_model

MAX_DIM = 10

accuracy = json.load(open('result_accuracy.json', 'r'))

def get_model_name(classifier_model, dim_model='', dimensions=0):
    dim_str = ''
    if dimensions:
        dim_str = "_" + str(dimensions)
    
    return f'tfidf_{dim_model}{dim_str}_{classifier_model}'

classification_model_names = list(classification_model.keys())
dims = list(range(2, MAX_DIM + 1))

# what to do
# plot the PCA vs SVD vs Normal Model
normal_model_acc = {}
for cname in classification_model_names:
    normal_model_acc[cname] = np.mean(accuracy[get_model_name(cname)])

pca_acc = {}
svd_acc = {}
for cname in classification_model_names:
    pca = []
    svd = []
    for dim in dims:
        # i love you [redacted]
        pca.append(np.mean(accuracy[get_model_name(cname, 'pca', dim)]))
        svd.append(np.mean(accuracy[get_model_name(cname, 'truncated_svd', dim)]))
    pca_acc[cname] = pca.copy()
    svd_acc[cname] = svd.copy()

def plot_acc(normal_model, pca, svd, classifier_model_name):
    plt.figure()
    plt.title(f'TFIDF vs PCA vs SVD of {classifier_model_name}')
    plt.xlabel('Dimensionality')
    plt.ylabel('Accuracy')
    plt.plot(dims, [normal_model] * len(dims), label='Plain TF-IDF')
    plt.plot(dims, pca, label='TFIDF-PCA')
    plt.plot(dims, svd, label='TFIDF-Truncated-SVD')
    plt.legend()
    plt.savefig(f'acc_plot/{classifier_model_name}.png')

for cname in classification_model_names:
    plot_acc(normal_model_acc[cname], pca_acc[cname], svd_acc[cname], cname)

# best model
all_models = {}
for mod, acc in accuracy.items():
    all_models[mod] = np.mean(acc)

print("Best Model According to Accuracy : {}".format(max(all_models, key=lambda k: all_models[k])))

dim_red_models = {k: v for k, v in all_models.items() if ('pca' in k or 'svd' in k)}

print("Best Dimensionality Reduction Model According to Accuracy : {}".format(max(dim_red_models, key=lambda k: dim_red_models[k])))

plt.show()