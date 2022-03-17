import pandas as pd

def get_news_dataset():

    news_dataset_dir = "news_dataset/"
    fake_news_dir = news_dataset_dir + "Fake.csv"
    true_news_dir = news_dataset_dir + "True.csv"

    fake_news = pd.read_csv(fake_news_dir)
    true_news = pd.read_csv(true_news_dir)

    content_column = 'text'

    fake_news_content = fake_news[content_column]
    true_news_content = true_news[content_column]

    return fake_news_content.values, true_news_content.values
