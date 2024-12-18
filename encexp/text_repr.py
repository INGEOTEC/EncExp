# Copyright 2024 Mario Graff (https://github.com/mgraffg)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Union
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from b4msa import TextModel
from microtc.utils import Counter
from microtc import emoticons
from microtc.weighting import TFIDF
import numpy as np
from numpy.linalg import norm
from encexp.download import download_seqtm, download_encexp
from encexp.utils import replace_tokens, progress_bar


class TM(TextModel):
    """TextModel where the vocabulary is obtained with SeqTM"""

    def __init__(self, lang='es',
                 voc_size_exponent: int=13,
                 vocabulary=None,
                 prefix_suffix: bool=True,
                 voc_source: str='mix',
                 precision=np.float32):
        if vocabulary is None:
            vocabulary = download_seqtm(lang, voc_source=voc_source,
                                        voc_size_exponent=voc_size_exponent,
                                        prefix=self.__class__.__name__.lower(),
                                        prefix_suffix=prefix_suffix)
        self._map = {}
        params = vocabulary['params']
        counter = vocabulary['counter']
        if not isinstance(counter, Counter):
            counter = Counter(counter["dict"],
                              counter["update_calls"])
        super().__init__(**params)
        self.language = lang
        self.voc_size_exponent = voc_size_exponent
        self._vocabulary(counter)
        self.prefix_suffix = prefix_suffix
        self.precision = precision
        replace_tokens(self)

    def fit(self, X, y):
        """fit"""
        return self

    def _vocabulary(self, counter):
        """Vocabulary"""

        tfidf = TFIDF()
        tfidf.N = counter.update_calls
        tfidf.word2id, tfidf.wordWeight = tfidf.counter2weight(counter)
        self.model = tfidf

    @property
    def language(self):
        """Language of the pre-trained text representations"""

        return self._language

    @language.setter
    def language(self, value):
        self._language = value

    @property
    def voc_size_exponent(self):
        """Vocabulary size :math:`2^v`; where :math:`v` is :py:attr:`voc_size_exponent` """

        return self._voc_size_exponent

    @voc_size_exponent.setter
    def voc_size_exponent(self, value):
        self._voc_size_exponent = value

    @property
    def identifier(self):
        """Function id"""

        lang = self.language
        voc = self.voc_size_exponent
        return f'seqtm_{lang}_{voc}'

    @property
    def names(self):
        """Vector space components"""

        try:
            return self._names
        except AttributeError:
            _names = [None] * len(self.id2token)
            for k, v in self.id2token.items():
                _names[k] = v
            self.names = np.array(_names)
            return self._names

    @names.setter
    def names(self, value):
        self._names = value

    @property
    def weights(self):
        """Vector space weights"""

        try:
            return self._weights
        except AttributeError:
            w = [None] * len(self.token_weight)
            for k, v in self.token_weight.items():
                w[k] = v
            self.weights = np.array(w)
            return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value


    def tonp(self, X):
        """Sparse representation to sparce matrix

        :param X: Sparse representation of matrix
        :type X: list
        :rtype: csr_matrix
        """
        from scipy.sparse import csr_matrix

        if not isinstance(X, list):
            return X
        assert self.num_terms is not None
        data = []
        row = []
        col = []
        for r, x in enumerate(X):
            col.extend([i for i, _ in x])
            data.extend([v for _, v in x])
            _ = [r] * len(x)
            row.extend(_)
        return csr_matrix((data, (row, col)),
                          shape=(len(X), self.num_terms),
                          dtype=self.precision)


class SeqTM(TM):
    """TextModel where the utterance is segmented in a sequence."""
    def _vocabulary(self, counter):
        """Vocabulary"""

        super(SeqTM, self)._vocabulary(counter)
        tfidf = self.model
        tokens = self.tokens
        for value in tfidf.word2id:
            key = value
            if value[:2] == 'q:':
                key = value[2:]
                if key in self._map:
                    continue
                self._map[key] = value
            else:
                key = f'~{key}~'
                self._map[key] = value
            tokens[key] = False    

    def compute_tokens(self, text):
        """
        Labels in a text

        :param text:
        :type text: str
        :returns: The labels in the text
        :rtype: set
        """

        get = self._map.get
        lst = self.find_token(text)
        _ = [text[a:b] for a, b in lst]
        return [[get(x, x) for x in _]]

    @property
    def tokens(self):
        """Tokens"""

        try:
            return self._tokens
        except AttributeError:
            self._tokens = OrderedDict()
        return self._tokens
    
    @tokens.setter
    def tokens(self, value):
        self._tokens = value

    @property
    def data_structure(self):
        """Datastructure"""

        try:
            return self._data_structure
        except AttributeError:
            _ = emoticons.create_data_structure
            self._data_structure = _(self.tokens)
        return self._data_structure

    @data_structure.setter
    def data_structure(self, value):
        self._data_structure = value

    def find_token(self, text):
        """Obtain the position of each label in the text

        :param text: text
        :type text: str
        :return: list of pairs, init and end of the word
        :rtype: list
        """

        blocks = []
        init = i = end = 0
        head = self.data_structure
        current = head
        text_length = len(text)
        while i < text_length:
            char = text[i]
            try:
                current = current[char]
                i += 1
                if '__end__' in current:
                    end = i
                    if current['__end__'] is True:
                        raise KeyError
            except KeyError:
                current = head
                if end > init:
                    blocks.append([init, end])
                    if (end - init) >= 2 and text[end - 1] == '~':
                        init = i = end = end - 1
                    else:
                        init = i = end
                elif i > init:
                    if (i - init) >= 2 and text[i - 1] == '~':
                        init = end = i = i - 1
                    else:
                        init = end = i
                else:
                    init += 1
                    i = end = init
        if end > init:
            blocks.append([init, end])
        return blocks


@dataclass
class EncExpT:
    """EncExpT (Encaje Explicable)
    
    Represent a text in the embedding using the `transform`method.
    """
    lang: str='es'
    voc_size_exponent: int=13
    EncExp_filename: str=None
    precision: np.dtype=np.float32
    voc_source: str='mix'
    enc_source: str=None
    prefix_suffix: bool=True
    merge_IDF: bool=True
    force_token: bool=True
    intercept: bool=False
    transform_distance: bool=False
    unit_vector: bool=True
    tailored: Union[bool, str]=False
    progress_bar: bool=False

    def get_params(self, deep=None):
        """Parameters"""
        return dict(lang=self.lang,
                    voc_size_exponent=self.voc_size_exponent,
                    EncExp_filename=self.EncExp_filename,
                    precision=self.precision,
                    voc_source=self.voc_source,
                    enc_source=self.enc_source,
                    prefix_suffix=self.prefix_suffix,
                    merge_IDF=self.merge_IDF,
                    force_token=self.force_token,
                    intercept=self.intercept,
                    transform_distance=self.transform_distance,
                    unit_vector=self.unit_vector,
                    tailored=self.tailored,
                    progress_bar=self.progress_bar)

    def set_params(self, **kwargs):
        """Set the parameters"""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def fit(self, D, y=None):
        """Estimate the parameters"""
        if self.tailored is not False:
            self.build_tailored(D, load=True)
        return self

    def force_tokens_weights(self, IDF: bool=False):
        """Set the maximum weight"""
        # rows = np.arange(len(self.names))
        rows = np.array([i for i, k in enumerate(self.names)
                         if k in self.bow.token2id])

        cols = np.array([self.bow.token2id[x] for x in self.names
                         if x in self.bow.token2id])
        if cols.shape[0] == 0:
            return
        if IDF:
            w = self.weights[rows][:, cols] * self.bow.weights[cols]
            _max = (w.max(axis=1) / self.bow.weights[cols]).astype(self.precision)
        else:
            _max = self.weights[rows].max(axis=1)
        self.weights[rows, cols] = _max

    @property
    def bias(self):
        """Bias / Intercept"""
        try:
            return self._bias
        except AttributeError:
            self.weights
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = value

    @property
    def weights(self):
        """Weights"""
        try:
            return self._weights
        except AttributeError:
            if self.EncExp_filename is not None:
                data = download_encexp(output=self.EncExp_filename)
            else:
                if self.intercept:
                    assert not self.merge_IDF
                data = download_encexp(lang=self.lang,
                                       voc_size_exponent=self.voc_size_exponent,
                                       voc_source=self.voc_source,
                                       enc_source=self.enc_source,
                                       prefix_suffix=self.prefix_suffix,
                                       intercept=self.intercept)
            self.bow = SeqTM(vocabulary=data['seqtm'])
            w = self.bow.weights
            weights = []
            precision = self.precision
            for vec in data['coefs']:
                if not self.merge_IDF:
                    coef = vec['coef']
                else:
                    coef = (vec['coef'] * w).astype(precision)
                weights.append(coef)
            self.weights = np.vstack(weights)
            self.bias = np.array([vec['intercept'] for vec in data['coefs']],
                                 dtype=self.precision)
            self.names = np.array([vec['label'] for vec in data['coefs']])
            self.enc_training_size = {vec['label']: vec['N'] for vec in data['coefs']}
            if self.force_token:
                self.force_tokens_weights(IDF=self.intercept)
        self.weights = np.asarray(self._weights, order='F')
        return self._weights

    @property
    def weights_norm(self):
        """Weights norm"""
        try:
            return self._weights_norm
        except AttributeError:
            _ = np.linalg.norm(self.weights, axis=1)
            self._weights_norm = _
        return self._weights_norm

    @property
    def enc_training_size(self):
        """Training size of each embedding"""
        try:
            return self._enc_training_size
        except AttributeError:
            self.weights
        return self._enc_training_size

    @enc_training_size.setter
    def enc_training_size(self, value):
        self._enc_training_size = value

    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def names(self):
        """Vector space components"""
        try:
            return self._names
        except AttributeError:
            self.weights
        return self._names

    @names.setter
    def names(self, value):
        self._names = value

    @property
    def bow(self):
        """BoW"""
        try:
            return self._bow
        except AttributeError:
            self.weights
        return self._bow

    @bow.setter
    def bow(self, value):
        self._bow = value

    def encode(self, text):
        """Encode utterace into a matrix"""

        token2id = self.bow.token2id
        seq = []
        for token in self.bow.tokenize(text):
            try:
                seq.append(token2id[token])
            except KeyError:
                continue
        W = self.weights
        if len(seq) == 0:
            return np.ones((W.shape[0], 1), dtype=W.dtype)
        return W[:, seq]

    def transform(self, texts):
        """Represents the texts into a matrix"""
        if self.intercept:
            X = self.bow.transform(texts) @ self.weights.T + self.bias
        else:
            X = np.r_[[self.encode(data).sum(axis=1)
                      for data in progress_bar(texts, total=len(texts),
                                               desc='Transform',
                                               use_tqdm=self.progress_bar)]]
        if self.transform_distance:
            X = X / self.weights_norm
        if self.unit_vector:
            _norm = norm(X, axis=1)
            _norm[_norm == 0] = 1
            return X / np.c_[_norm]
        return X

    def fill(self, inplace: bool=True, names: list=None):
        """Fill weights with the missing dimensions"""
        weights = self.weights
        if names is None:
            names = self.bow.names
        w = np.zeros((len(names), weights.shape[1]),
                     dtype=self.precision)
        iden = {v: k for k, v in enumerate(names)}
        for key, value in zip(self.names, weights):
            w[iden[key]] = value
        if inplace:
            self.weights = w
            self.names = names
        return w

    def build_tailored(self, data, load=False, **kwargs):
        """Build a tailored model with data"""

        import os
        from os.path import isfile
        from tempfile import mkstemp
        from json import dumps
        from microtc.utils import tweet_iterator
        from encexp.download import download_seqtm
        from encexp.build_encexp import build_encexp
        if hasattr(self, '_tailored_built'):
            return None

        get_text = self.bow.get_text
        if isinstance(self.tailored, str) and isfile(self.tailored):
            if load:
                _ = self.__class__(EncExp_filename=self.tailored)
                self.__iadd__(_)
                self._tailored_built = True
            return None
        iden, path = mkstemp()
        with open(iden, 'w', encoding='utf-8') as fpt:
            for d in data:
                print(dumps(dict(text=get_text(d))), file=fpt)
        if isinstance(self.tailored, bool):
            _, self.tailored = mkstemp(suffix='.gz')
        if self.EncExp_filename is not None:
            voc = next(tweet_iterator(self.EncExp_filename))
        else:
            voc = download_seqtm(self.lang, self.voc_size_exponent,
                                 voc_source=self.voc_source)
        build_kw = dict(min_pos=16, tokens=self.names)
        build_kw.update(kwargs)
        build_encexp(voc, path, self.tailored, **build_kw)
        os.unlink(path)
        if load:
            self.__iadd__(self.__class__(EncExp_filename=self.tailored))
            self._tailored_built = True

    def __add__(self, other):
        """Add weights"""
        ins = clone(self)
        return ins.__iadd__(other)

    def __iadd__(self, other):
        """Add weights"""

        assert np.all(self.bow.names == other.bow.names)
        _ = self.precision == np.float32
        weights_ = self.weights if _ else self.weights.astype(np.float32)
        _ = other.precision == np.float32
        w_other = other.weights if _ else other.weights.astype(np.float32)
        w_norm = np.linalg.norm(weights_, axis=1)
        other_norm = np.linalg.norm(w_other, axis=1)
        w = dict(zip(self.names, weights_ / np.c_[w_norm]))
        w_other = dict(zip(other.names, w_other / np.c_[other_norm]))
        w_norm = dict(zip(self.names, w_norm))
        other_norm = dict(zip(other.names, other_norm))
        names = sorted(set(self.names).union(set(other.names)))
        weights = []
        norms = []
        for name in names:
            if name in w and name in w_other:
                _ = (w[name] + w_other[name]) / 2
                weights.append(_)
                norms.append(w_norm[name])
            elif name in w:
                weights.append(w[name])
                norms.append(w_norm[name])
            else:
                weights.append(w_other[name])
                norms.append(other_norm[name])
        weights = np.asarray(weights, order='F')
        weights = weights / np.c_[np.linalg.norm(weights, axis=1)]
        self.weights = np.asarray(weights * np.c_[np.array(norms)],
                                  dtype=self.precision, order='F')
        self.names = np.array(names)
        return self

    def __sklearn_clone__(self):
        klass = self.__class__
        params = self.get_params()
        ins = klass(**params)
        ins.weights = self.weights
        ins.bow = self.bow
        ins.names = self.names
        ins.enc_training_size = self.enc_training_size
        if hasattr(self, '_tailored_built'):
            ins._tailored_built = self._tailored_built
        return ins


@dataclass
class EncExp(EncExpT):
    """EncExp (Encaje Explicable)"""

    estimator_kwargs: dict=None
    kfold_class: StratifiedKFold=StratifiedKFold
    kfold_kwargs: dict=None

    def get_params(self, deep=None):
        """Parameters"""
        params = super(EncExp, self).get_params()
        params.update(dict(estimator_kwargs=self.estimator_kwargs,
                           kfold_class=self.kfold_class,
                           kfold_kwargs=self.kfold_kwargs))
        return params

    def fit(self, D, y=None):
        """Estimate the parameters"""
        super(EncExp, self).fit(D, y=y)
        if y is None:
            y = [x['klass'] for x in D]
        if not hasattr(self, '_estimator') and len(D) > 2**17:
            self.estimator = SGDClassifier(class_weight='balanced')
        X = self.transform(D)
        self.estimator.fit(X, y)
        return self
    
    @property
    def estimator(self):
        """Estimator (classifier/regressor)"""
        try:
            return self._estimator
        except AttributeError:
            from sklearn.svm import LinearSVC
            params = dict(class_weight='balanced',
                          dual='auto')
            if self.estimator_kwargs is not None:
                params.update(self.estimator_kwargs)
            self.estimator_kwargs = params
            self.estimator = LinearSVC(**self.estimator_kwargs)
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

    def predict(self, texts):
        """Predict"""
        X = self.transform(texts)
        return self.estimator.predict(X)

    def decision_function(self, texts):
        """Decision function"""
        X = self.transform(texts)
        hy = self.estimator.decision_function(X)
        if hy.ndim == 1:
            return np.c_[hy]
        return hy

    def train_predict_decision_function(self, D, y=None):
        """Train and predict the decision"""
        if y is None:
            y = np.array([x['klass'] for x in D])
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        nclass = np.unique(y).shape[0]
        X = self.transform(D)
        if nclass == 2:
            hy = np.empty(X.shape[0])
        else:
            hy = np.empty((X.shape[0], nclass))
        kwargs = dict(random_state=0, shuffle=True)
        if self.kfold_kwargs is not None:
            kwargs.update(self.kfold_kwargs)
        for tr, vs in self.kfold_class(**kwargs).split(X, y):
            m = clone(self).estimator.fit(X[tr], y[tr])
            hy[vs] = m.decision_function(X[vs])
        if hy.ndim == 1:
            return np.c_[hy]
        return hy

    def __sklearn_clone__(self):
        ins = super(EncExp, self).__sklearn_clone__()
        if hasattr(self, '_estimator'):
            ins.estimator = clone(self.estimator)
        return ins
