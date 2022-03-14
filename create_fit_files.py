import os
import spacy
import pandas as pd

from src.feature_extractor.tf_idf_vectorizer import TfidfVectorizer
from src.feature_extractor.count_vectorizer import CountVectorizer
from src import vsr_nlp


def main():
    nlp = spacy.load("pt_core_news_sm")
    tokenizer = nlp.tokenizer

    path = "./data/split_into_subdocs"

    chunks_sizes = [128, 256, 512]

    for chunks_size in chunks_sizes:
        df = pd.read_csv(os.path.join(path, f'corpus_chunks_{chunks_size}.csv'), sep=';')
        df.drop("Unnamed: 0", axis=1, inplace=True)

        text_corpus = df['text'].astype(str).apply(vsr_nlp.normalize).values.astype(str)

        feature_extractor = CountVectorizer(tokenizer=tokenizer)
        feature_extractor.fit(text_corpus)
        feature_extractor.save_fit(chunks_size)

        feature_extractor = TfidfVectorizer(tokenizer=tokenizer)
        feature_extractor.fit(text_corpus)
        feature_extractor.save_fit(chunks_size)


if __name__ == "__main__":
    main()
