import pandas as pd
from tqdm.auto import tqdm
import os

# load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/Ajblahblah-01/tweet_dataset_labelled_by_roberta/main/labelled_dataset.csv')

def add_sentiment_token(text, sentiment):
    if sentiment == 'Negative':
        return '<neg> ' + text
    elif sentiment == 'Positive':
        return '<pos> ' + text
    elif sentiment == 'Neutral':
        return '<neu> ' + text
    else:
        return text

df['Tweet'] = df.apply(lambda x: add_sentiment_token(x['Tweet'], x['sentiment']), axis=1)

# generate samples for tokenizer
PATH = './content'
paths = []

text_data = []
file_count = 0

for sample in tqdm(df['Tweet'].values):
    text_data.append(sample)
    if len(text_data) == 10000:
        with open(os.path.join(PATH, f'text_data_{file_count}.txt'), 'w') as f:
            f.write('/n'.join(text_data))
        file_count += 1
        text_data = []

# remaining samples to be written to file
with open(os.path.join(PATH, f'text_data_{file_count}.txt'), 'w') as f:
    f.write('/n'.join(text_data))