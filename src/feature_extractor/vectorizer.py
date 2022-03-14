import numpy as np


class Vectorizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.unique_tokens = None
        self.c_matrix = None
        self.counts = None
        self.vocab = None

    def fit(self, corpus):
        raise NotImplementedError

    def transform(self, corpus):
        raise NotImplementedError

    def normalize_token(self, text):
        # Remove espaços não necessários
        # e.g. "  eu" => "eu"
        return ' '.join(text.split())

    def tokenize(self, doc):
        tokens = []
        for t in self.tokenizer(str(doc)):
            tokens.append(self.normalize_token(str(t)))
        return tokens

    def create_vocab(self, corpus):
        all_tokens = []
        for doc in corpus:
            # tokenize retorna uma lista de tokens
            tokens = self.tokenize(doc)

            # OBS: tokens é uma lista
            all_tokens.extend(tokens)

        self.unique_tokens, self.counts = np.unique(all_tokens, return_counts=True)
        self.vocab = np.array({w: i for i, w in enumerate(self.unique_tokens)})

    def save_fit(self, chunks_size):
        raise NotImplementedError

    def load_fit(self, chunks_size):
        raise NotImplementedError
