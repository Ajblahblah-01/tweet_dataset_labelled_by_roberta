from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./arpit')

print(tokenizer('[POS] this stock is looking good').input_ids)