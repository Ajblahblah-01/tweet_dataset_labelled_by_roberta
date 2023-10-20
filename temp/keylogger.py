import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing the dataset
df = pd.read_csv('tweet_dataset.csv')

print(df.dtypes)
# compile all the tweets from above dataset into one paragraph
paragraph = df['Tweet'].str.cat(sep='\n')

print(paragraph[:100])


