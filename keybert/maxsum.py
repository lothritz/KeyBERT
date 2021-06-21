import numpy as np
import itertools
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from keybert.apsyn import APSyn

def max_sum_similarity(doc_embedding: np.ndarray,
                       word_embeddings: np.ndarray,
                       words: List[str],
                       top_n: int,
                       nr_candidates: int,
                       sm:str="cosine") -> List[Tuple[str, float]]:
    """ Calculate Max Sum Distance for extraction of keywords

    We take the 2 x top_n most similar words/phrases to the document.
    Then, we take all top_n combinations from the 2 x top_n words and
    extract the combination that are the least similar to each other
    by cosine similarity.

    NOTE:
        This is O(n^2) and therefore not advised if you use a large top_n

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        nr_candidates: The number of candidates to consider

    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances
    """
    if nr_candidates < top_n:
        raise Exception("Make sure that the number of candidates exceeds the number "
                        "of keywords to return.")

    # Calculate distances and extract keywords
    if sm=="cosine":
       distances = cosine_similarity(doc_embedding, word_embeddings)
       distances_words = cosine_similarity(word_embeddings, word_embeddings)
       words_idx = list(distances.argsort()[0][-nr_candidates:])
    elif sm=="apsyn":
       doc_embedding_ref=[(1,i,doc_embedding[0][i]) for i in range(len(doc_embedding[0]))]
       candidate_embeddings_ref=[[(1,i,word_embeddings[j][i]) for i in range(len(word_embeddings[j]))] for j in range(len(word_embeddings))]
       distances=np.array([APSyn(doc_embedding_ref,candidate_embeddings_ref[i])[0] for i in range(len(candidate_embeddings_ref))])
       distances_words=np.array([np.array([APSyn(candidate_embeddings_ref[i], candidate_embeddings_ref[j])[0] for j in range(len(candidate_embeddings_ref))]) for i in range(len(candidate_embeddings_ref))])
       words_idx = list(distances.argsort()[-nr_candidates:])
    # Get 2*top_n words as candidates based on cosine similarity
    
    words_vals = [words[index] for index in words_idx]
    candidates = distances_words[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = 100_000
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim
    if sm=="cosine":
       return [(words_vals[idx], round(float(distances[0][idx]), 4)) for idx in candidate]
    elif sm=="apsyn":
       return [(words_vals[idx], round(float(distances[idx]), 4)) for idx in candidate]
