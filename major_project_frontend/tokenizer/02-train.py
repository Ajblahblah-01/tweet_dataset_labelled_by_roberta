import os
from pathlib import Path
from tokenizers import BertWordPieceTokenizer

paths = [str(x) for x in Path('./content').glob('**/*.txt')]


# initialize a tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=False,
)

# and then train
tokenizer.train(files = paths, vocab_size=5_000, min_frequency=2, show_progress=True, wordpieces_prefix='##' , special_tokens=[
                    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]' , '[POS]', '[NEG]', '[NEU]'])

tokenizer.save_model('./arpit', 'arpit')