import os
import time
import numpy as np
import spacy
import string
import unidecode
import pandas as pd
from sklearn.metrics import accuracy_score

from src.feature_extractor.count_vectorizer import CountVectorizer
from src.feature_extractor.tf_idf_vectorizer import TfidfVectorizer
from src.feature_extractor.bm25 import BM25
from src import vsr_nlp


def extract_aula(txt):
    num = int(float(txt.split('aula')[1]))
    return num


def main():
    nlp = spacy.load("pt_core_news_sm")
    tokenizer = nlp.tokenizer

    chunk_size = 512

    df = pd.read_csv(f'data/split_into_subdocs/corpus_chunks_{chunk_size}.csv', sep=';')

    df['normalized text'] = df['text'].astype(str).apply(vsr_nlp.normalize)
    text_corpus = df['normalized text'].values.astype(str)

    df_test = pd.read_csv('data/test_dataset.csv')
    df_test['pergunta normalizada'] = df_test['Pergunta'].astype(str).apply(vsr_nlp.normalize)
    questions_corpus = df_test['pergunta normalizada'].values.astype(str)

    feature_extractor = TfidfVectorizer(tokenizer=tokenizer)
    X = feature_extractor.load_fit(chunk_size)

    Q = feature_extractor.transform(questions_corpus)
    i_opt = vsr_nlp.cos_distance(X, Q)

    retrieved_documents = df.iloc[i_opt, :]
    #
    predicted_course = retrieved_documents['curso'].values
    predicted_aula = retrieved_documents['aula'].apply(extract_aula).values
    #
    ground_truth_course = df_test['Curso'].values.astype(str)
    ground_truth_aula = df_test['Aula'].values.astype(int)

    acc_course = 100 * sum(1. * (predicted_course == ground_truth_course)) / len(df_test)
    #
    acc_aula = 100 * sum(
        1 * np.logical_and(predicted_course == ground_truth_course, predicted_aula == ground_truth_aula)) / len(df_test)
    #
    print(acc_course, acc_aula)


if __name__ == "__main__":
    main()
