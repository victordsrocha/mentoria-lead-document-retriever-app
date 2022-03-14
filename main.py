import pandas as pd
from fastapi import FastAPI
import spacy
from src.feature_extractor.tf_idf_vectorizer import TfidfVectorizer
from src.document_retriever import DocumentRetriever

app = FastAPI()

# Global Variables
df = pd.read_csv('data/split_into_subdocs/corpus_chunks_128.csv', sep=';')
# feature_extractor = TF(path='./data/')

# df.drop("Unnamed: 0", axis=1, inplace=True)
nlp = spacy.load("pt_core_news_sm")
tokenizer = nlp.tokenizer

feature_extractor = TfidfVectorizer(tokenizer=tokenizer)
document_retriever = DocumentRetriever(feature_extractor, chunks_size=128, dataframe=df)


# Functions
@app.get("/")
async def readme():
    return "Colocar instruções de execução aqui"


@app.get("/get/{student_question}")
async def get_context(student_question: str):
    text, course, aula = document_retriever.retrieve_document(student_question)
    response = f'Olá! A resposta para a sua pergunta pode ser encontrada no curso {course}, aula {aula}'

    return response
