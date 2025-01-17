## **Version 0.3.0**
*Release date:  10 May, 2021*  

The two main features are **candidate keywords** 
and several **backends** to use instead of Flair and SentenceTransformers!

**Highlights**:

* Use candidate words instead of extracting those from the documents ([#25](https://github.com/MaartenGr/KeyBERT/issues/25))
    * ```KeyBERT().extract_keywords(doc, candidates)```
* Spacy, Gensim, USE, and Custom Backends were added (see documentation [here](https://maartengr.github.io/KeyBERT/guides/embeddings.html))

**Fixes**:

* Improved imports
* Fix encoding error when locally installing KeyBERT ([#30](https://github.com/MaartenGr/KeyBERT/issues/30)) 

**Miscellaneous**:

* Improved documentation (ReadMe & MKDocs)
* Add the main tutorial as a shield
* Typos ([#31](https://github.com/MaartenGr/KeyBERT/pull/31), [#35](https://github.com/MaartenGr/KeyBERT/pull/35))


## **Version 0.2.0**
*Release date:  9 Feb, 2021*  

**Highlights**:

* Add similarity scores to the output
* Add Flair as a possible back-end
* Update documentation + improved testing

## **Version 0.1.2*
*Release date:  28 Oct, 2020*  

Added Max Sum Similarity as an option to diversify your results.


## **Version 0.1.0**
*Release date:  27 Oct, 2020*  

This first release includes keyword/keyphrase extraction using BERT and simple cosine similarity. 
There is also an option to use Maximal Marginal Relevance to select the candidate keywords/keyphrases.
