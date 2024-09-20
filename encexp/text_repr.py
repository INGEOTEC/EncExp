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
from b4msa import TextModel
from microtc.utils import tweet_iterator, Counter
from microtc import emoticons
from microtc.weighting import TFIDF
import numpy as np
from encexp.download import download_seqtm, download_encexp


class SeqTM(TextModel):
    """TextModel where the utterance is segmented in a sequence."""

    def __init__(self, lang='es',
                 voc_size_exponent: int=13,
                 vocabulary=None):
        if vocabulary is None:
            vocabulary = download_seqtm(lang,
                                        voc_size_exponent=voc_size_exponent)
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


@dataclass
class EncExp:
    lang: str='es'
    voc_size_exponent: int=13
    EncExp_filename: str=None
    precision: np.dtype=np.float32

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
                                       precision=self.precision)
            self._bow = SeqTM(vocabulary=data['seqtm'])
            w = self._bow.weights
            weights = []
            precision = self.precision
            for vec in data['coefs']:
                coef = (vec['coef'] * w).astype(precision)
                _ = coef.max()
                coef[self._bow.token2id[vec['label']]] = _
                weights.append(coef)
            self._weights = np.vstack(weights)
            self._names = np.array([vec['label'] for vec in data['coefs']])
        return self._weights

    @property
    def names(self):
        """Vector space components"""
        try:
            return self._names
        except AttributeError:
            self.weights
        return self._names
    
    @property
    def bow(self):
        """BoW"""
        try:
            return self._bow
        except AttributeError:
            self.weights
        return self._bow

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
        return np.vstack([W[:, x] for x in seq]).T

    def transform(self, texts):
        """Represents the texts into a matrix"""
        enc = []
        for data in texts:
            _ = self.encode(data)
            vec = _.sum(axis=1)
            enc.append(vec / np.linalg.norm(vec))
        return np.vstack(enc)        