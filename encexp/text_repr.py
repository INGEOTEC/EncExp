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
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from b4msa import TextModel
from microtc.utils import tweet_iterator, Counter
from microtc import emoticons
from microtc.weighting import TFIDF
import numpy as np
from numpy.linalg import norm
from encexp.download import download_seqtm, download_encexp


class SeqTM(TextModel):
    """TextModel where the utterance is segmented in a sequence."""

    def __init__(self, lang='es',
                 voc_size_exponent: int=13,
                 vocabulary=None,
                 prefix_suffix: bool=True,
                 precision=np.float32):
        if vocabulary is None:
            vocabulary = download_seqtm(lang,
                                        voc_size_exponent=voc_size_exponent,
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
        self.__vocabulary(counter)
        self.prefix_suffix = prefix_suffix
        self.precision = precision

    def __vocabulary(self, counter):
        """Vocabulary"""

        from os.path import join, dirname
        tfidf = TFIDF()
        tfidf.N = counter.update_calls
        tfidf.word2id, tfidf.wordWeight = tfidf.counter2weight(counter)
        self.model = tfidf
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
            tokens[key] = value
        _ = join(dirname(__file__), 'data', 'emojis.json.gz')
        emojis = next(tweet_iterator(_))
        for k, v in emojis.items():
            self._map[k] = v
            tokens[k] = v
            for x in [f'~{k}~', f'~{k}', f'{k}~']:
                self._map[x] = v
                tokens[x] = v

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
    def sequence(self):
        """Vocabulary compute on sequence text-transformation"""

        return self._sequence

    @sequence.setter
    def sequence(self, value):
        self._sequence = value

    @property
    def names(self):
        """Vector space components"""

        try:
            return self._names
        except AttributeError:
            _names = [None] * len(self.id2token)
            for k, v in self.id2token.items():
                _names[k] = v
            self._names = np.array(_names)
            return self._names

    @property
    def weights(self):
        """Vector space weights"""

        try:
            return self._weights
        except AttributeError:
            w = [None] * len(self.token_weight)
            for k, v in self.token_weight.items():
                w[k] = v
            self._weights = np.array(w)
            return self._weights

    @property
    def tokens(self):
        """Tokens"""

        try:
            return self._tokens
        except AttributeError:
            self._tokens = OrderedDict()
        return self._tokens

    @property
    def data_structure(self):
        """Datastructure"""

        try:
            return self._data_structure
        except AttributeError:
            _ = emoticons.create_data_structure
            self._data_structure = _(self.tokens)
        return self._data_structure

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
                if "__end__" in current:
                    end = i
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

@dataclass
class EncExp:
    """EncExp (Encaje Explicable)"""
    lang: str='es'
    voc_size_exponent: int=13
    EncExp_filename: str=None
    precision: np.dtype=np.float16
    country: str=None
    prefix_suffix: bool=True
    estimator_kwargs: dict=None
    merge_IDF: bool=True
    force_token: bool=True
    kfold_class: StratifiedKFold=StratifiedKFold
    kfold_kwargs: dict=None
    intercept: bool=False

    def get_params(self):
        """Parameters"""
        return dict(lang=self.lang,
                    voc_size_exponent=self.voc_size_exponent,
                    EncExp_filename=self.EncExp_filename,
                    precision=self.precision,
                    country=self.country,
                    prefix_suffix=self.prefix_suffix,
                    estimator_kwargs=self.estimator_kwargs,
                    merge_IDF=self.merge_IDF,
                    force_token=self.force_token,
                    kfold_class=self.kfold_class,
                    kfold_kwargs=self.kfold_kwargs)

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

    def fit(self, D, y=None):
        """Estimate the parameters"""
        if y is None:
            y = [x['klass'] for x in D]
        X = self.transform(D)
        self.estimator.fit(X, y)
        return self

    def force_tokens_weights(self, IDF: bool=False):
        """Set the maximum weight"""
        rows = np.arange(len(self.names))
        cols = np.array([self.bow.token2id[x] for x in self.names])
        w = self.weights
        if IDF:
            w = w * self.bow.weights
            _max = (w.max(axis=1) / self.bow.weights).astype(self.precision)
        else:
            _max = w.max(axis=1)
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
                data = download_encexp(output=self.EncExp_filename,
                                       precision=self.precision)
            else:
                data = download_encexp(lang=self.lang,
                                       voc_size_exponent=self.voc_size_exponent,
                                       precision=self.precision,
                                       country=self.country,
                                       prefix_suffix=self.prefix_suffix)
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
            if self.force_token:
                self.force_tokens_weights()
        if self.intercept:
            self.weights = np.asarray(self._weights, order='F')
        return self._weights

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
        flag = self.weights.dtype == np.float16
        if self.intercept:
            X = self.bow.transform(texts) @ self.weights.T + self.bias
        else:
            X = np.r_[[self.encode(data).sum(axis=1)
                    for data in texts]]
        if flag:
            X = X.astype(np.float32)
        _norm = norm(X, axis=1)
        _norm[_norm == 0] = 1
        return X / np.c_[_norm]

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

    def fill(self, inplace: bool=True):
        """Fill weights with the missing dimensions"""
        weights = self.weights
        w = np.zeros((len(self.bow.names), weights.shape[1]),
                     dtype=self.precision)
        iden = {v:k for k, v in enumerate(self.bow.names)}
        for key, value in zip(self.names, weights):
            w[iden[key]] = value
        if inplace:
            self.weights = w
            self.names = self.bow.names
        return w

    def __sklearn_clone__(self):
        klass = self.__class__
        params = self.get_params()
        ins = klass(**params)
        ins.weights = self.weights
        ins.bow = self.bow
        ins.names = self.names
        return ins
