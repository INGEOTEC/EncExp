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
from abc import ABC
import inspect
from typing import Union, Iterable
from collections import OrderedDict
from unicodedata import normalize
import re
from os.path import isfile
import os
import numpy as np
from numpy.linalg import norm
from sklearn.base import clone
from microtc.params import OPTION_NONE, OPTION_GROUP
from microtc.utils import Counter
from microtc import emoticons, TextModel as microTCTM
from microtc.textmodel import SKIP_SYMBOLS
from microtc.weighting import TFIDF
from encexp.download import download
from encexp.utils import progress_bar


class Identifier(ABC):
    """Identifier"""
    def get_params(self, deep=False):
        """TextModel parameters"""
        sig = inspect.signature(self.__class__)
        params = sorted(sig.parameters.keys())
        return {k: getattr(self, k) for k in params}
    
    def identifier_filter(self, key, value):
        """Test default parameters"""
        return False

    @property
    def identifier(self):
        """Identifier - parameters md5"""
        import hashlib
        sig = inspect.signature(self.__class__)
        diff = []
        for k, v in sig.parameters.items():
            value = getattr(self, k)
            if self.identifier_filter(k, value):
                continue
            if value == v.default:
                continue
            diff.append([k, value])
        diff.sort(key=lambda x: x[0])
        cdn = ' '.join([f'{k}={v}'
                        for k, v in diff])
        _ = hashlib.md5(bytes(cdn,
                              encoding='utf-8')).hexdigest()
        return f'{self.__class__.__name__}_{_}'

    @property
    def precision(self):
        """precision"""
        try:
            return self._precision
        except AttributeError:
            return np.float32

    @precision.setter
    def precision(self, value):
        self._precision = value

    def download(self, first: bool=True):
        """download"""
        return download(self.identifier, first=first)


class TextModel(Identifier, microTCTM):
    """TextModel"""

    def __init__(self, lang: str=None, text: str='text',
                 num_option: str=OPTION_NONE, usr_option: str=OPTION_GROUP,
                 url_option: str=OPTION_GROUP, emo_option: str=OPTION_NONE,
                 hashtag_option: str=OPTION_NONE, ent_option: str=OPTION_NONE,
                 lc: bool=True, del_dup: bool=False, del_punc: bool=False,
                 del_diac: bool=True, select_ent: bool=False, select_suff: bool=False,
                 select_conn: bool=False, max_dimension: bool=True,
                 unit_vector: bool=True, q_grams_words: bool=True,
                 norm_emojis: bool=True, token_list: list=None,
                 token_min_filter: Union[int, float]=0,
                 token_max_filter: Union[int, float]=int(2**17),
                 weighting: str='microtc.weighting.TFIDF',
                 norm_punc: bool=True, pretrained=True):
        if token_list is None:
            if lang in ['ja', 'zh']:
                token_list = [1, 2, 3]
            else:
                token_list = [-2, -1, 2, 3, 4]
        super().__init__(text=text, num_option=num_option, usr_option=usr_option,
                         url_option=url_option, emo_option=emo_option,
                         hashtag_option=hashtag_option, ent_option=ent_option,
                         lc=lc, del_dup=del_dup, del_punc=del_punc, del_diac=del_diac,
                         select_ent=select_ent, select_suff=select_suff,
                         select_conn=select_conn, max_dimension=max_dimension, 
                         unit_vector=unit_vector, q_grams_words=q_grams_words,
                         norm_emojis=False, token_list=token_list,
                         token_min_filter=token_min_filter,
                         token_max_filter=token_max_filter, weighting=weighting)
        self.text = self._text
        self.lang = lang
        self.norm_emojis = norm_emojis
        assert norm_punc | del_punc
        if norm_punc:
            assert norm_emojis
        self.norm_punc = norm_punc
        self._norm_tokens()
        self.pretrained = pretrained
        if pretrained:
            counter = self.download()['vocabulary']
            self.set_vocabulary(counter)

    def identifier_filter(self, key, value):
        """Test default parameters"""
        if key == 'token_list':
            if self.lang in ('ja', 'zh'):
                tk_lst = [1, 2, 3]
            else:
                tk_lst = [-2, -1, 2, 3, 4]
            if tk_lst == value:
                return True
        return False

    def set_vocabulary(self, counter: Counter):
        """Set vocabulary"""

        if not isinstance(counter, Counter):
            counter = Counter(counter["dict"],
                              counter["update_calls"])
        tfidf = TFIDF()
        tfidf.N = counter.update_calls
        tfidf.word2id, tfidf.wordWeight = tfidf.counter2weight(counter)
        self.model = tfidf
        self.pretrained = True

    def fit(self, X, y=None):
        """Estimate the tokens weights"""
        if self.pretrained:
            return self
        super().fit(X)
        return self

    def _norm_tokens(self):
        """Normalize tokens"""
        _ = ['_htag', '_ent', '_num', '_url', '_usr']
        self.norm_tokens = {k: f'~e:{k}~' for k in _}
        if self.norm_emojis:
            _ = {k:f'~e:{v.replace("~", "")}~'
                 for k, v in emoticons.read_emojis().items()}
            self.norm_tokens.update(_)
        if self.norm_punc:
            _ = {k: f'~e:{k}~' for k in SKIP_SYMBOLS if k != '~'}
            self.norm_tokens.update(_)
        _ = {x: True for x in self.norm_tokens}
        self.norm_head = emoticons.create_data_structure(_)

    def text_transformations(self, text:str):
        """Text transformations

        :param text: text
        :type text: str

        :rtype: str
        """

        text = super(TextModel, self).text_transformations(text)
        text = re.sub('~+', '~', text)
        if self.del_diac:
            return text
        return normalize('NFD', text)

    def get_word_list(self, text):
        """Words from normalize text"""
        data = text.split('~')
        return data[1:-1]

    def compute_q_grams_words(self, textlist):
        """q-grams only on words"""
        output = []
        textlist = ['~' + x + '~' for x in textlist if x[:2] != 'e:']
        for qsize in self.q_grams:
            _ = qsize - 1
            extra = [x for x in textlist if len(x) >= _]
            qgrams = [["".join(output) for output in zip(*[text[i:] for i in range(qsize)])] 
                      for text in extra]
            for _ in qgrams:
                for x in _:
                    output.append("q:" + x)
        return output

    def compute_q_grams(self, text):
        """q-grams"""
        output = []
        inner = []
        for word in self.get_word_list(text):
            if word[:2] == 'e:':
                if len(inner) > 0:
                    output.extend(self.compute_q_grams_words(['~'.join(inner)]))
                    inner = []
                continue
            inner.append(word)
        if len(inner) > 0 :
            output.extend(self.compute_q_grams_words(['~'.join(inner)]))
        return output

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
            self.weights = np.array(w, dtype=self.precision)
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


class SeqTM(TextModel):
    """TextModel where the utterance is segmented in a sequence."""
    def __init__(self, lang: str=None, text: str='text',
                 num_option: str=OPTION_NONE, usr_option: str=OPTION_GROUP,
                 url_option: str=OPTION_GROUP, emo_option: str=OPTION_NONE,
                 hashtag_option: str=OPTION_NONE, ent_option: str=OPTION_NONE,
                 lc: bool=True, del_dup: bool=False, del_punc: bool=False,
                 del_diac: bool=True, select_ent: bool=False, select_suff: bool=False,
                 select_conn: bool=False, max_dimension: bool=True,
                 unit_vector: bool=True, q_grams_words: bool=True,
                 norm_emojis: bool=True, token_list: list=None,
                 token_min_filter: Union[int, float]=0,
                 token_max_filter: Union[int, float]=int(2**14),
                 weighting: str='microtc.weighting.TFIDF',
                 norm_punc: bool=True, pretrained=True):
        super().__init__(lang=lang,
                         text=text, num_option=num_option, usr_option=usr_option,
                         url_option=url_option, emo_option=emo_option,
                         hashtag_option=hashtag_option, ent_option=ent_option,
                         lc=lc, del_dup=del_dup, del_punc=del_punc, del_diac=del_diac,
                         select_ent=select_ent, select_suff=select_suff,
                         select_conn=select_conn, max_dimension=max_dimension,
                         unit_vector=unit_vector, q_grams_words=q_grams_words,
                         norm_emojis=norm_emojis, token_list=token_list,
                         token_min_filter=token_min_filter,
                         token_max_filter=token_max_filter, weighting=weighting,
                         norm_punc=norm_punc, pretrained=pretrained)

    @property
    def token_id(self):
        """Token id"""
        try:
            return self._token_id
        except AttributeError:
            self.token_id = {}
            return self._token_id

    @token_id.setter
    def token_id(self, value):
        self._token_id = value

    def set_vocabulary(self, counter: Counter):
        """Set vocabulary"""

        super().set_vocabulary(counter)
        tfidf = self.model
        tokens = OrderedDict()
        code = {}
        for value in tfidf.word2id:
            key = value
            if value[:2] == 'q:':
                key = value[2:]
                if key in code:
                    continue
                code[key] = value
            else:
                key = f'~{key}~'
                code[key] = value
            tokens[key] = False
        self.tokens = tokens
        self.token_id = code
        if self.norm_punc or self.norm_emojis:
            if '~e:' not in self.tokens and '~e:~' in self.tokens:
                self.tokens['~e:'] = False
                self.token_id['~e:'] = self.token_id['~e:~']

    def compute_tokens(self, text):
        """
        Labels in a text

        :param text:
        :type text: str
        :returns: The labels in the text
        :rtype: set
        """

        get = self.token_id.get
        lst = self.find_token(text)
        _ = [text[a:b] for a, b in lst]
        return [[get(x, x) for x in _]]

    @property
    def tokens(self):
        """Tokens"""

        try:
            return self._tokens
        except AttributeError:
            self.tokens = OrderedDict()
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
class EncExpT(Identifier):
    """EncExp Transformation"""
    lang: str=None
    token_max_filter: int=int(2**14)
    pretrained: bool=True
    use_tqdm: bool=True
    with_intercept: bool=False
    merge_encode: bool=True
    distance: bool=False
    keep_unfreq: bool=False

    @property
    def seqTM(self):
        """SeqTM"""
        try:
            return self._seqTM
        except AttributeError:
            _ = SeqTM(lang=self.lang,
                      token_max_filter=self.token_max_filter)
            self.seqTM = _
        return self._seqTM
    
    def identifier_filter(self, key, value):
        """Test default parameters"""
        if key == 'use_tqdm':
            return True
        if key == 'pretrained':
            return True
        if key == 'merge_encode':
            return True
        if key == 'distance':
            return True
        return False

    @seqTM.setter
    def seqTM(self, value):
        self._seqTM = value

    @property
    def weights(self):
        """Weights"""
        try:
            return self._weights
        except AttributeError:
            assert self.pretrained
            self.set_weights(self.download(first=False))
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def _names_weights_intercept(self, data: Iterable):
        """Get names, weights and intercept from data"""
        weights = []
        names = []
        intercept = []
        for coef in data:
            _ = np.frombuffer(bytearray.fromhex(coef['coef']),
                              dtype=np.float16)
            weights.append(_)
            label = coef['label']
            if isinstance(label, list):
                names.extend(label)
            else:
                names.append(coef['label'])
            if self.with_intercept:
                _ = np.frombuffer(bytearray.fromhex(coef['intercept']),
                                  dtype=np.float16)
                intercept.append(_[0])
        if self.with_intercept:
            intercept = np.asanyarray(intercept,
                                      dtype=self.precision)
        _ = np.column_stack(weights)
        return np.array(names), np.asanyarray(_, dtype=self.precision), intercept

    def set_weights(self, data: Iterable):
        """Set weights"""
        self.names, self.weights, intercept = self._names_weights_intercept(data)
        if self.with_intercept:
            self.intercept = intercept
            
    @property
    def intercept(self):
        """Intercept"""
        return self._intercept

    @intercept.setter
    def intercept(self, value):
        self._intercept = value

    @property
    def names(self):
        """Component names"""
        return self._names

    @names.setter
    def names(self, value):
        self._names = value

    def encode(self, text):
        """Encode utterace into a matrix"""

        token2id = self.seqTM.token2id
        seq = []
        for token in self.seqTM.tokenize(text):
            try:
                seq.append(token2id[token])
            except KeyError:
                continue
        W = self.weights
        tfidf = self.seqTM.weights
        if len(seq) == 0:
            return np.ones((1, W.shape[1]), dtype=W.dtype)
        index, tf_ = np.unique(seq, return_counts=True)
        # cnt = Counter(seq)
        # seq = np.array(list(cnt.keys()))
        # tf = np.array([cnt[k] for k in seq])
        tf = tf_ / tf_.sum()
        _ = tfidf[index] * tf
        if self.merge_encode:
            return W[index] * np.c_[_ / norm(_)]
        tfidf = {k: v for k, v in zip(index, _ / (norm(_) * tf_))}
        return W[seq] * np.c_[[tfidf[i] for i in seq]]
        
    def transform(self, texts: Iterable):
        """Transform"""
        _ = [self.encode(text).sum(axis=0) 
             for text in progress_bar(texts, desc='Transform',
                                      use_tqdm=self.use_tqdm)]
        X = np.vstack(_)
        if self.with_intercept:
            X = X + self.intercept
        if self.distance:
            X = X / self.norm
        return X

    def fit(self, X, y):
        """fit"""
        return self

    @property
    def norm(self):
        """Weights norm"""
        try:
            return self._norm
        except AttributeError:
            _ = np.linalg.norm(self.weights, axis=0)
            self.norm = _
        return self._norm

    @norm.setter
    def norm(self, value):
        self._norm = value

    def add(self, data: Iterable):
        """Add weights"""
        names, weights, _ = self._names_weights_intercept(data)
        assert isinstance(_, list)
        self_name_w = dict(zip(self.names, self.weights.T))
        name_w = dict(zip(names, weights.T))
        names = sorted(set(self.names).union(set(names)))
        zeros = np.zeros(self.weights.shape[0])
        weights = []
        for name in names:
            frst = 1 if name in self_name_w else 0
            scnd = 1 if name in name_w else 0
            frst = frst / (frst + scnd)
            v1 = self_name_w.get(name, zeros)
            v2 = name_w.get(name, zeros)
            weights.append(frst * v1 + (1 - frst) * v2)
        self.weights = np.column_stack(weights)
        self.names = names
    
    def tailored(self, D: Iterable=None,
                 filename: str=None,
                 tsv_filename: str=None,
                 min_pos: int=512,
                 min_neg: int=int(2**14),
                 max_pos: int=int(2**14),
                 n_jobs: int=-1,
                 self_supervised: bool=True,
                 ds: object=None,
                 train: object=None):
        """Load/Create tailored encexp representation"""
        from tempfile import mkstemp
        from microtc.utils import tweet_iterator
        from encexp.build_encexp import EncExpDataset, Train

        def weights(args):
            for fname, _ in args:
                data = next(tweet_iterator(fname))
                yield data

        def set_weights(data):
            if self.pretrained:
                return self.add(data)
            self.set_weights(data)

        if self.pretrained:
            _ = self.weights
        if filename is not None:
            filename = filename.split('.json.gz')[0]
        if filename is not None and isfile(f'{filename}.json.gz'):
            set_weights(tweet_iterator(f'{filename}.json.gz'))
            return self
        if ds is None:
            ds = EncExpDataset(text_model=clone(self.seqTM),
                               self_supervised=self_supervised,
                               use_tqdm=self.use_tqdm)
        if tsv_filename is None:
            _, path = mkstemp()
        else:
            path = tsv_filename
        ds.output_filename = path
        if tsv_filename is None or not isfile(tsv_filename):
            ds.process(D)
        if train is None:
            train = Train(text_model=clone(self.seqTM),
                          filename=ds.output_filename,
                          use_tqdm=self.use_tqdm,
                          min_pos=min_pos,
                          min_neg=min_neg,
                          max_pos=max_pos,
                          n_jobs=n_jobs,
                          with_intercept=self.with_intercept,
                          self_supervised=self_supervised,
                          keep_unfreq=self.keep_unfreq)
        if filename is None:
            train.identifier = self.identifier
        else:
            train.identifier = filename
        if filename is None:
            args = train.create_model()
            set_weights(weights(args))
            train.delete_tmps(args)
        else:
            train.store_model()
            set_weights(tweet_iterator(f'{train.identifier}.json.gz'))
        if tsv_filename is None:
            try:
                os.unlink(path)
            except PermissionError:
                pass
        return self


# @dataclass
# class EncExpT:
#     """EncExpT (Encaje Explicable)
    
#     Represent a text in the embedding using the `transform`method.
#     """
#     lang: str='es'
#     voc_size_exponent: int=13
#     EncExp_filename: str=None
#     precision: np.dtype=np.float32
#     voc_source: str='mix'
#     enc_source: str=None
#     prefix_suffix: bool=True
#     merge_IDF: bool=True
#     force_token: bool=True
#     intercept: bool=False
#     transform_distance: bool=False
#     unit_vector: bool=True
#     tailored: Union[bool, str]=False
#     progress_bar: bool=False

#     def get_params(self, deep=None):
#         """Parameters"""
#         return dict(lang=self.lang,
#                     voc_size_exponent=self.voc_size_exponent,
#                     EncExp_filename=self.EncExp_filename,
#                     precision=self.precision,
#                     voc_source=self.voc_source,
#                     enc_source=self.enc_source,
#                     prefix_suffix=self.prefix_suffix,
#                     merge_IDF=self.merge_IDF,
#                     force_token=self.force_token,
#                     intercept=self.intercept,
#                     transform_distance=self.transform_distance,
#                     unit_vector=self.unit_vector,
#                     tailored=self.tailored,
#                     progress_bar=self.progress_bar)

#     def set_params(self, **kwargs):
#         """Set the parameters"""
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#     def fit(self, D, y=None):
#         """Estimate the parameters"""
#         if self.tailored is not False:
#             self.build_tailored(D, load=True)
#         return self

#     def force_tokens_weights(self, IDF: bool=False):
#         """Set the maximum weight"""
#         # rows = np.arange(len(self.names))
#         rows = np.array([i for i, k in enumerate(self.names)
#                          if k in self.bow.token2id])

#         cols = np.array([self.bow.token2id[x] for x in self.names
#                          if x in self.bow.token2id])
#         if cols.shape[0] == 0:
#             return
#         if IDF:
#             w = self.weights[rows][:, cols] * self.bow.weights[cols]
#             _max = (w.max(axis=1) / self.bow.weights[cols]).astype(self.precision)
#         else:
#             _max = self.weights[rows].max(axis=1)
#         self.weights[rows, cols] = _max

#     @property
#     def bias(self):
#         """Bias / Intercept"""
#         try:
#             return self._bias
#         except AttributeError:
#             self.weights
#         return self._bias

#     @bias.setter
#     def bias(self, value):
#         self._bias = value

#     @property
#     def weights(self):
#         """Weights"""
#         try:
#             return self._weights
#         except AttributeError:
#             if self.EncExp_filename is not None:
#                 data = download_encexp(output=self.EncExp_filename)
#             else:
#                 if self.intercept:
#                     assert not self.merge_IDF
#                 data = download_encexp(lang=self.lang,
#                                        voc_size_exponent=self.voc_size_exponent,
#                                        voc_source=self.voc_source,
#                                        enc_source=self.enc_source,
#                                        prefix_suffix=self.prefix_suffix,
#                                        intercept=self.intercept)
#             self.bow = SeqTM(vocabulary=data['seqtm'])
#             w = self.bow.weights
#             weights = []
#             precision = self.precision
#             for vec in data['coefs']:
#                 if not self.merge_IDF:
#                     coef = vec['coef']
#                 else:
#                     coef = (vec['coef'] * w).astype(precision)
#                 weights.append(coef)
#             self.weights = np.vstack(weights)
#             self.bias = np.array([vec['intercept'] for vec in data['coefs']],
#                                  dtype=self.precision)
#             self.names = np.array([vec['label'] for vec in data['coefs']])
#             self.enc_training_size = {vec['label']: vec['N'] for vec in data['coefs']}
#             if self.force_token:
#                 self.force_tokens_weights(IDF=self.intercept)
#         self.weights = np.asarray(self._weights, order='F')
#         return self._weights

#     @property
#     def weights_norm(self):
#         """Weights norm"""
#         try:
#             return self._weights_norm
#         except AttributeError:
#             _ = np.linalg.norm(self.weights, axis=1)
#             self._weights_norm = _
#         return self._weights_norm

#     @property
#     def enc_training_size(self):
#         """Training size of each embedding"""
#         try:
#             return self._enc_training_size
#         except AttributeError:
#             self.weights
#         return self._enc_training_size

#     @enc_training_size.setter
#     def enc_training_size(self, value):
#         self._enc_training_size = value

#     @weights.setter
#     def weights(self, value):
#         self._weights = value

#     @property
#     def names(self):
#         """Vector space components"""
#         try:
#             return self._names
#         except AttributeError:
#             self.weights
#         return self._names

#     @names.setter
#     def names(self, value):
#         self._names = value

#     @property
#     def bow(self):
#         """BoW"""
#         try:
#             return self._bow
#         except AttributeError:
#             self.weights
#         return self._bow

#     @bow.setter
#     def bow(self, value):
#         self._bow = value

#     def encode(self, text):
#         """Encode utterace into a matrix"""

#         token2id = self.bow.token2id
#         seq = []
#         for token in self.bow.tokenize(text):
#             try:
#                 seq.append(token2id[token])
#             except KeyError:
#                 continue
#         W = self.weights
#         if len(seq) == 0:
#             return np.ones((W.shape[0], 1), dtype=W.dtype)
#         return W[:, seq]

#     def transform(self, texts):
#         """Represents the texts into a matrix"""
#         if self.intercept:
#             X = self.bow.transform(texts) @ self.weights.T + self.bias
#         else:
#             X = np.r_[[self.encode(data).sum(axis=1)
#                       for data in progress_bar(texts, total=len(texts),
#                                                desc='Transform',
#                                                use_tqdm=self.progress_bar)]]
#         if self.transform_distance:
#             X = X / self.weights_norm
#         if self.unit_vector:
#             _norm = norm(X, axis=1)
#             _norm[_norm == 0] = 1
#             return X / np.c_[_norm]
#         return X

#     def fill(self, inplace: bool=True, names: list=None):
#         """Fill weights with the missing dimensions"""
#         weights = self.weights
#         if names is None:
#             names = self.bow.names
#         w = np.zeros((len(names), weights.shape[1]),
#                      dtype=self.precision)
#         iden = {v: k for k, v in enumerate(names)}
#         for key, value in zip(self.names, weights):
#             w[iden[key]] = value
#         if inplace:
#             self.weights = w
#             self.names = names
#         return w

#     def build_tailored(self, data, load=False, **kwargs):
#         """Build a tailored model with data"""

#         import os
#         from os.path import isfile
#         from tempfile import mkstemp
#         from json import dumps
#         from microtc.utils import tweet_iterator
#         from encexp.download import download_seqtm
#         from encexp.build_encexp import build_encexp
#         if hasattr(self, '_tailored_built'):
#             return None

#         get_text = self.bow.get_text
#         if isinstance(self.tailored, str) and isfile(self.tailored):
#             if load:
#                 _ = self.__class__(EncExp_filename=self.tailored)
#                 self.__iadd__(_)
#                 self._tailored_built = True
#             return None
#         iden, path = mkstemp()
#         with open(iden, 'w', encoding='utf-8') as fpt:
#             for d in data:
#                 print(dumps(dict(text=get_text(d))), file=fpt)
#         if isinstance(self.tailored, bool):
#             _, self.tailored = mkstemp(suffix='.gz')
#         if self.EncExp_filename is not None:
#             voc = next(tweet_iterator(self.EncExp_filename))
#         else:
#             voc = download_seqtm(self.lang, self.voc_size_exponent,
#                                  voc_source=self.voc_source)
#         build_kw = dict(min_pos=16, tokens=self.names)
#         build_kw.update(kwargs)
#         build_encexp(voc, path, self.tailored, **build_kw)
#         os.unlink(path)
#         if load:
#             self.__iadd__(self.__class__(EncExp_filename=self.tailored))
#             self._tailored_built = True

#     def __add__(self, other):
#         """Add weights"""
#         ins = clone(self)
#         return ins.__iadd__(other)

#     def __iadd__(self, other):
#         """Add weights"""

#         assert np.all(self.bow.names == other.bow.names)
#         _ = self.precision == np.float32
#         weights_ = self.weights if _ else self.weights.astype(np.float32)
#         _ = other.precision == np.float32
#         w_other = other.weights if _ else other.weights.astype(np.float32)
#         w_norm = np.linalg.norm(weights_, axis=1)
#         other_norm = np.linalg.norm(w_other, axis=1)
#         w = dict(zip(self.names, weights_ / np.c_[w_norm]))
#         w_other = dict(zip(other.names, w_other / np.c_[other_norm]))
#         w_norm = dict(zip(self.names, w_norm))
#         other_norm = dict(zip(other.names, other_norm))
#         names = sorted(set(self.names).union(set(other.names)))
#         weights = []
#         norms = []
#         for name in names:
#             if name in w and name in w_other:
#                 _ = (w[name] + w_other[name]) / 2
#                 weights.append(_)
#                 norms.append(w_norm[name])
#             elif name in w:
#                 weights.append(w[name])
#                 norms.append(w_norm[name])
#             else:
#                 weights.append(w_other[name])
#                 norms.append(other_norm[name])
#         weights = np.asarray(weights, order='F')
#         weights = weights / np.c_[np.linalg.norm(weights, axis=1)]
#         self.weights = np.asarray(weights * np.c_[np.array(norms)],
#                                   dtype=self.precision, order='F')
#         self.names = np.array(names)
#         return self

#     def __sklearn_clone__(self):
#         klass = self.__class__
#         params = self.get_params()
#         ins = klass(**params)
#         ins.weights = self.weights
#         ins.bow = self.bow
#         ins.names = self.names
#         ins.enc_training_size = self.enc_training_size
#         if hasattr(self, '_tailored_built'):
#             ins._tailored_built = self._tailored_built
#         return ins


# @dataclass
# class EncExp(EncExpT):
#     """EncExp (Encaje Explicable)"""

#     estimator_kwargs: dict=None
#     kfold_class: StratifiedKFold=StratifiedKFold
#     kfold_kwargs: dict=None

#     def get_params(self, deep=None):
#         """Parameters"""
#         params = super(EncExp, self).get_params()
#         params.update(dict(estimator_kwargs=self.estimator_kwargs,
#                            kfold_class=self.kfold_class,
#                            kfold_kwargs=self.kfold_kwargs))
#         return params

#     def fit(self, D, y=None):
#         """Estimate the parameters"""
#         super(EncExp, self).fit(D, y=y)
#         if y is None:
#             y = [x['klass'] for x in D]
#         if not hasattr(self, '_estimator') and len(D) > 2**17:
#             self.estimator = SGDClassifier(class_weight='balanced')
#         X = self.transform(D)
#         self.estimator.fit(X, y)
#         return self
    
#     @property
#     def estimator(self):
#         """Estimator (classifier/regressor)"""
#         try:
#             return self._estimator
#         except AttributeError:
#             from sklearn.svm import LinearSVC
#             params = dict(class_weight='balanced',
#                           dual='auto')
#             if self.estimator_kwargs is not None:
#                 params.update(self.estimator_kwargs)
#             self.estimator_kwargs = params
#             self.estimator = LinearSVC(**self.estimator_kwargs)
#         return self._estimator

#     @estimator.setter
#     def estimator(self, value):
#         self._estimator = value

#     def predict(self, texts):
#         """Predict"""
#         X = self.transform(texts)
#         return self.estimator.predict(X)

#     def decision_function(self, texts):
#         """Decision function"""
#         X = self.transform(texts)
#         hy = self.estimator.decision_function(X)
#         if hy.ndim == 1:
#             return np.c_[hy]
#         return hy

#     def train_predict_decision_function(self, D, y=None):
#         """Train and predict the decision"""
#         if y is None:
#             y = np.array([x['klass'] for x in D])
#         if not isinstance(y, np.ndarray):
#             y = np.array(y)
#         nclass = np.unique(y).shape[0]
#         X = self.transform(D)
#         if nclass == 2:
#             hy = np.empty(X.shape[0])
#         else:
#             hy = np.empty((X.shape[0], nclass))
#         kwargs = dict(random_state=0, shuffle=True)
#         if self.kfold_kwargs is not None:
#             kwargs.update(self.kfold_kwargs)
#         for tr, vs in self.kfold_class(**kwargs).split(X, y):
#             m = clone(self).estimator.fit(X[tr], y[tr])
#             hy[vs] = m.decision_function(X[vs])
#         if hy.ndim == 1:
#             return np.c_[hy]
#         return hy

#     def __sklearn_clone__(self):
#         ins = super(EncExp, self).__sklearn_clone__()
#         if hasattr(self, '_estimator'):
#             ins.estimator = clone(self.estimator)
#         return ins
