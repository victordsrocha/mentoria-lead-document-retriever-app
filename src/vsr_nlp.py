import string
import unidecode
import numpy as np
from scipy.spatial.distance import cdist


def normalize(text):
    # Passa os carácteres para minúsculo
    preprocessed = text.lower()
    # Remove acentos
    preprocessed = unidecode.unidecode(preprocessed)
    # Muda os separadores de sentença (não necessário)
    preprocessed = preprocessed.replace(". ", "[SEP]")
    preprocessed = preprocessed.replace(".", " ")
    preprocessed = preprocessed.replace("[SEP]", ". ")
    preprocessed = preprocessed.replace("-", " ")
    # Remove pontuação (a verificar para o caso de programação)
    preprocessed = preprocessed.translate(str.maketrans("", "", string.punctuation))

    return preprocessed


def cos_distance(x_matrix, q_matrix):
    m = cdist(x_matrix, q_matrix, metric='euclidean')
    i_opt = m.argmin(axis=0)
    return i_opt
