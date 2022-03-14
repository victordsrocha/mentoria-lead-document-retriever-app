import numpy as np
from src.feature_extractor.vectorizer import Vectorizer


class TfidfVectorizer(Vectorizer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self.idf = None

    def fit(self, corpus):
        self.create_vocab(corpus)

        tf = self.calc_tf(corpus)
        df = np.sum(tf.astype(bool), axis=0)
        self.idf = np.log(len(corpus) / (df + 1))

        self.c_matrix = tf * self.idf
        return self.c_matrix

    def transform(self, question_corpus):
        tf = self.calc_tf(question_corpus)
        return tf * self.idf

    def calc_tf(self, corpus):
        tf = []
        for doc in corpus:
            tokens = self.tokenize(doc)

            x = np.zeros([len(self.vocab.item())])
            u, c = np.unique(tokens, return_counts=True)
            c = c / sum(c)

            for ui, ci in zip(u, c):
                if len(ui) < 2:
                    continue
                try:
                    index = self.vocab.item()[ui]
                    x[index] = ci
                except KeyError:
                    pass

            tf.append(x)
        return np.squeeze(np.array(tf))

    def save_fit(self, chunks_size):
        np.savez('./data/saved_features/fit_file_tfidf_vectorizer_{}.npz'.format(chunks_size),
                 v=np.array(self.vocab),
                 u=self.unique_tokens,
                 c=self.counts,
                 i=self.idf,
                 m=self.c_matrix)

    def load_fit(self, chunks_size):
        load = np.load('./data/saved_features/fit_file_tfidf_vectorizer_{}.npz'.format(chunks_size), allow_pickle=True)
        self.vocab = load['v']
        self.unique_tokens = load['u']
        self.counts = load['c']
        self.idf = load['i']
        self.c_matrix = load['m']
        load.close()
        return self.c_matrix
