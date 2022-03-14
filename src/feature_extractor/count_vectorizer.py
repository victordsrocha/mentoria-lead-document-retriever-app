import numpy as np
from src.feature_extractor.vectorizer import Vectorizer


class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def fit(self, corpus):
        self.create_vocab(corpus)
        self.c_matrix = self.transform(corpus)
        return self.c_matrix

    def transform(self, corpus):
        v_matrix = []
        for doc in corpus:
            tokens = self.tokenize(doc)
            x = self.vectorize(tokens)
            v_matrix.append(x)
        v_matrix = np.squeeze(np.array(v_matrix))
        return v_matrix

    def vectorize(self, tokens):
        vec = np.zeros([len(self.vocab.item())])
        u, c = np.unique(tokens, return_counts=True)

        for ui, ci in zip(u, c):
            if len(ui) < 2:
                continue
            try:
                index = self.vocab.item()[ui]
                vec[index] = ci
            except KeyError:
                pass

        return vec

    def save_fit(self, chunks_size):
        np.savez('./data/saved_features/fit_file_count_vectorizer_{}.npz'.format(chunks_size),
                 v=np.array(self.vocab),
                 u=self.unique_tokens,
                 c=self.counts,
                 m=self.c_matrix)

    def load_fit(self, chunks_size):
        load = np.load('./data/saved_features/fit_file_count_vectorizer_{}.npz'.format(chunks_size), allow_pickle=True)
        self.vocab = load['v']
        self.unique_tokens = load['u']
        self.counts = load['c']
        self.c_matrix = load['m']
        load.close()
        return self.c_matrix
