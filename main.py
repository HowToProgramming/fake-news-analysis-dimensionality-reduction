import numpy as np
import time
import json
from model import create_dim_red_visualize_models, create_models, evaluate_model
from dataset import get_news_dataset
import warnings


warnings.filterwarnings('ignore')
t = time.time()

fake_news, real_news = get_news_dataset()

# Change Data Limit to 0 if we're going to go full model
data_limit = 100
# Limit the data (for low RAM capacity purpose)
if data_limit:
    fake_news = fake_news[:data_limit]
    real_news = real_news[:data_limit]

documents = np.concatenate((fake_news, real_news))
labels = np.concatenate((np.zeros(len(fake_news)), np.ones(len(real_news))))

models = create_models()
eval_scores = {}
for model_lbl, model in models.items():
    print(f'Evaluating : {model_lbl}')
    eval_scores[model_lbl] = list(evaluate_model(model, documents, labels))
    print(f'Acc : {np.mean(eval_scores[model_lbl])} | Stdev : {np.std(eval_scores[model_lbl])}')

pca, svd = create_dim_red_visualize_models()
pca.fit_transform(documents)
svd.fit_transform(documents)

with open('result_accuracy.json', 'w+') as acc_fp:
    json.dump(eval_scores, acc_fp)

print("Processing Successful")
print("Runtime :", time.time() - t)