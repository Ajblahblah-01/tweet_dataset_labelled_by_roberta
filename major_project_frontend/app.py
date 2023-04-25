from flask import Flask, render_template, request
import torch
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertTokenizer

from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from scipy.special import softmax

from tqdm import tqdm

device = ( "cuda" if torch.cuda.is_available() else "cpu")

import re

# url = 'https://raw.githubusercontent.com/Ajblahblah-01/tweet_dataset_labelled_by_roberta/main/labelled_dataset.csv'
# df = pd.read_csv(url)
# def add_sentiment_token(text, sentiment):
#     if sentiment == 'Negative':
#         return '<neg> ' + text
#     elif sentiment == 'Positive':
#         return '<pos> ' + text
#     elif sentiment == 'Neutral':
#         return '<neu> ' + text
#     else:
#         return text
# df['Tweet'] = df.apply(lambda x: add_sentiment_token(x['Tweet'], x['sentiment']), axis=1)
# text = ''.join(df['Tweet'].tolist())
# text = re.sub(r'[^a-zA-Z0-9\s\!\@\#\$\%\^\&\*\(\)\_\+\-\=\[\]\{\}\|\;\:\'\"\,\.\/\<\>\?]', '', text)
# chars = sorted(list(set(text)))
# chars_to_remove = ['\xa0', '|', '\t', '\n', '\r']
# translation_table = str.maketrans("", "", "".join(chars_to_remove))
# text = text.translate(translation_table)
# chars = sorted(list(set(text)))
# vocab_size = len(chars)

# stoi = { ch:i for i,ch in enumerate(chars) }
# itos = { i:ch for i,ch in enumerate(chars) }
# encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
# decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# data = torch.tensor(encode(text), dtype=torch.long)
# n = int(0.95*len(data)) # first 95% will be training, rest validation data
# train_data = data[:n]
# val_data = data[n:]

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 128 # what is the maximum context length for predictions?
max_iters = 5000 # number of steps
eval_interval = 100 # after how many iterations do we want the loss update
learning_rate = 1e-3 
eval_iters = 200 # while calculating loss
n_embd = 256 # embedding dimensio
n_head = 16 # number of heads in the multiattention
n_layer = 12 # number of decoder blocks
dropout = 0.1 

# def get_batch(split):
#     # generate a small batch of data of inputs x and targets y
#     data = train_data if split == 'train' else val_data
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i+block_size] for i in ix])
#     y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#     x, y = x.to(device), y.to(device)
#     return x, y

# @torch.no_grad()
# def estimate_loss():
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(eval_iters)
#         for k in range(eval_iters):
#             X, Y = get_batch(split)
#             logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        
        self.key = nn.Linear(n_embd, head_size, bias=False) # input is embedding-dimention = 64 and output is head_size = 4
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        # compute attention scores

        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) <= (block_size , batch_size , n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



tokenizer_bert = BertTokenizer.from_pretrained('./tokenizer/arpit')


def encode(s):
    # Tokenize the input string on the specified device
    tokens = tokenizer_bert.encode(s, add_special_tokens=True, return_tensors='pt').to(device)
    # Flatten the tokens to a 1D tensor
    return tokens.view(-1)

def decode(l):
     # Convert the list of integers to a 1D PyTorch tensor
    tokens = torch.tensor(l, dtype=torch.long, device=device)
    # Convert the tensor to a list of tokens
    tokens = tokens.view(-1)
    # Convert the list of tokens to a string
    return tokenizer_bert.decode(tokens)

roberta = "cardiffnlp/twitter-roberta-base-sentiment"
roberta_model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

def sent(tweet, max_length=350):
    # Preprocess tweet
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    tweet_proc = " ".join(tweet_words)

    # Truncate tweet to specified maximum length
    tweet_trunc = tweet_proc[:max_length]

    # Tokenize and encode tweet
    encoded_tweet = tokenizer(tweet_trunc, return_tensors='pt')

    # Perform sentiment analysis
    output = roberta_model(**encoded_tweet)
    scores = output[0][0].detach().cpu().numpy()
    scores = torch.softmax(torch.from_numpy(scores), dim=0)

    # Get the predicted sentiment label
    max_score, sentiment = torch.max(scores, dim=0)
    sentiment_label = labels[sentiment.item()]

    return sentiment_label

app = Flask(__name__)

# Load the PyTorch model
model = torch.load("/Users/arpit/Downloads/python_nb/major_project_frontend/05-model_12m.pth" , map_location=torch.device('cpu'))
model.eval()


# Define the home page route and the function to handle the user input
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def generate_text():
    # Get the user input from the text area
    input_text = request.form["input_text"]
    sentiment = ""
    if input_text == '<neu>': 
        sentiment = 'Neutral' 
    elif input_text == '<pos>': 
        sentiment = 'Positive' 
    elif input_text == '<neg>':
        sentiment = 'Negative'

    context = encode(input_text).reshape(1,-1)
    output_text = ""
    while True:
        output_text = decode(model.generate(context, max_new_tokens=150)[0].tolist())
        output_text = output_text.replace("< neg >", "").replace("< pos >", "").replace("< neu >", "").replace("[CLS]", "").replace("[SEP]", "")
        output_text = " ".join(output_text.split())
        output_text = output_text.strip()
        if len(sentiment) == 0 or sentiment == sent(output_text):
            break
    return render_template("index.html", output_text=output_text)

if __name__ == "__main__":
    app.run(debug=True)


# context_arr = ['<neg>' , '<neu>' , '<pos>']
# output_text = []
# for input_text in tqdm(context_arr):
    

#     context = encode(input_text).reshape(1,-1)
#     # context = torch.tensor(input, dtype=torch.long, device=device)
#     i = 0
#     while i < 3:
#         temp_text = decode(model.generate(context, max_new_tokens=150)[0].tolist())
#         temp_text = temp_text.replace("<neg>", "").replace("<pos>", "").replace("<neu>", "").replace("[CLS]", "").replace("[SEP]", "")
#         temp_text = " ".join(temp_text.split())
#         temp_text = temp_text.strip()
#         if sent(temp_text) == sentiment:
#             output_text.append(temp_text)
#             i += 1



# for text in output_text:
#     print(sent(text))