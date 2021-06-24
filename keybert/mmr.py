import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from keybert.apsyn import APSyn


def mmr(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: List[str],
        top_n: int = 5,
        diversity: float = 0.8,
        sm:str="cosine") -> List[Tuple[str, float]]:
    """ Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.


    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.

    Arguments:
        doc_embedding: The document embeddings
        word_embeddings: The embeddings of the selected candidate keywords/phrases
        words: The selected candidate keywords/keyphrases
        top_n: The number of keywords/keyhprases to return
        diversity: How diverse the select keywords/keyphrases are.
                   Values between 0 and 1 with 0 being not diverse at all
                   and 1 being most diverse.

    Returns:
         List[Tuple[str, float]]: The selected keywords/keyphrases with their distances

    """

    # Extract similarity within words, and between words and the document
    if sm=="cosine":
       word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
       word_similarity = cosine_similarity(word_embeddings)
    elif sm=="apsyn":
       doc_embedding_ref=[(1,i,doc_embedding[0][i]) for i in range(len(doc_embedding[0]))]
       candidate_embeddings_ref=[[(1,i,word_embeddings[j][i]) for i in range(len(word_embeddings[j]))] for j in range(len(word_embeddings))]
       word_doc_similarity = np.array([[x] for x in [APSyn(doc_embedding_ref,candidate_embeddings_ref[i])[0] for i in range(len(candidate_embeddings_ref))]])
       word_similarity = np.array([np.array([APSyn(candidate_embeddings_ref[i], candidate_embeddings_ref[j])[0] for j in range(len(candidate_embeddings_ref))]) for i in range(len(candidate_embeddings_ref))])

    # Initialize candidates and already choose best keyword/keyphras
    #print(len(word_doc_similarity))
    #print(len(word_doc_similarity[0]))
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [(words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4)) for idx in keywords_idx]

