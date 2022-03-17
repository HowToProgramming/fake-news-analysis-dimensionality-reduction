import numpy as np
import matplotlib.pyplot as plt

from pca import TfIdf_PCA, plot_news
from dataset import get_news_dataset

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