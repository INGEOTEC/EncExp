EncExp (Encaje Explicable)
====================================

.. image:: https://github.com/INGEOTEC/EncExp/actions/workflows/test.yaml/badge.svg
	:target: https://github.com/INGEOTEC/EncExp/actions/workflows/test.yaml

.. image:: https://badge.fury.io/py/EncExp.svg
	:target: https://badge.fury.io/py/EncExp

.. image:: https://coveralls.io/repos/github/INGEOTEC/EncExp/badge.svg?branch=develop
    :target: https://coveralls.io/github/INGEOTEC/EncExp?branch=develop

EncExp is a set of tools for creating and using explainable embeddings. As with any embedding, the aim is to have a set of vectors that can be associated with tokens, and consequently, an utterance can be represented in the vector space span by the vectors. However, the difference concerning the embedding estimated with GloVe or Word2Vec, among others, is that EncExp associates vectors where each component has a meaning. The component's value indicates whether the word associated with the component might be present in the sentence. 

The component's meaning is a direct consequence of the procedure used to estimate the embedding. EncExp estimates the embedding by solving $d$ binary self-supervised classification problems, where the label is the presence of a particular token. The classifier used is a linear Support Vector Machine. 