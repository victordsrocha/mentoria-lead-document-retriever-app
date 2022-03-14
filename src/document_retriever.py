import numpy as np
from scipy.spatial.distance import cdist
from src import vsr_nlp


class DocumentRetriever:
    def __init__(self, feature_extractor, chunks_size, dataframe):
        """

        :param feature_extractor:
        :param chunks_size:
        :param dataframe:
        """
        self.X = feature_extractor.load_fit(chunks_size)
        self.feature_extractor = feature_extractor
        self.df = dataframe

    def retrieve_document(self, query):
        """

        :param query:
        :return:
        """
        query = vsr_nlp.normalize(query)
        q = self.feature_extractor.transform(np.array([query]))
        distances = cdist(self.X, np.array([q]), metric='cosine')
        doc_idx = np.argmin(distances)
        best_doc = self.df.iloc[doc_idx]
        return best_doc.text, best_doc.curso, best_doc.aula
