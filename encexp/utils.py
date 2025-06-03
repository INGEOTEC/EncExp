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
from os.path import isfile, join, dirname, isdir
import os
from zipfile import ZipFile
from typing import Union
from urllib import request
from urllib.error import HTTPError
try:
    USE_TQDM = True
    from tqdm import tqdm
except ImportError:
    USE_TQDM = False
import numpy as np
from microtc.utils import tweet_iterator
MODEL_LANG = ['ar', 'ca', 'de', 'en', 'es', 'fr',
              'hi', 'in', 'it', 'ja', 'ko', 'nl',
              'pl', 'pt', 'ru', 'tl', 'tr', 'zh']


DialectID_URL = 'https://github.com/INGEOTEC/dialectid/releases/download/data'
EncExp_URL = 'https://github.com/INGEOTEC/EncExp/releases/download/data'
MODELS = join(dirname(__file__),
              'models')

class Download(object):
    """Download
    
    >>> from EncExp.utils import Download
    >>> d = Download("http://github.com", "t.html")
    """

    def __init__(self, url,
                 output='t.tmp',
                 use_tqdm: bool=True) -> None:
        self._url = url
        self._output = output
        if not USE_TQDM or not use_tqdm:
            self._use_tqdm = False
        else:
            self._use_tqdm = True
        try:
            request.urlretrieve(url, output, reporthook=self.progress)
        except HTTPError as exc:
            self._use_tqdm = False
            self.close()
            raise RuntimeError(f'URL=> {url}') from exc
        self.close()

    @property
    def tqdm(self):
        """tqdm"""

        if not self._use_tqdm:
            return None
        try:
            return self._tqdm
        except AttributeError:
            self.tqdm = tqdm(total=self._nblocks,
                             leave=False, desc=self._output)
        return self._tqdm
    
    @tqdm.setter
    def tqdm(self, value):
        self._tqdm = value

    def close(self):
        """Close tqdm if used"""
        if self._use_tqdm:
            self.tqdm.close()

    def update(self):
        """Update tqdm if used"""
        if self._use_tqdm:
            self.tqdm.update()

    def progress(self, nblocks, block_size, total):
        """tqdm progress"""

        self._nblocks = total // block_size
        self.update()


def progress_bar(data, total=np.inf,
                 use_tqdm: bool=True,
                 **kwargs):
    """Progress bar"""

    if not USE_TQDM or not use_tqdm:
        return data
    if total == np.inf:
        total = None
    return tqdm(data, total=total, **kwargs)


def uniform_sample(N, avail_data):
    """Uniform sample from the available data"""
    remaining = avail_data.copy()
    M = 0
    while M < N:
        index = np.where(remaining > 0)[0]
        if index.shape[0] == 0:
            break
        sample = np.random.randint(index.shape[0], size=N - M)
        sample_i, sample_cnt = np.unique(index[sample], return_counts=True)
        remaining[sample_i] = remaining[sample_i] - sample_cnt
        remaining[remaining < 0] = 0
        M = (avail_data - remaining).sum()
    return avail_data - remaining


def unit_length(data):
    """Convert EncExp weights to have unit length"""
    if data.dtype == np.float16:
        data = data.astype(np.float32)
    w = np.linalg.norm(data, axis=0)
    w[w == 0] = 1
    return data / w


def set_to_zero(data, percentage: float=0.95):
    """Set elements to zero"""
    if percentage == 1:
        data[data < 0] = 0
        return data
    ss = np.argsort(data, axis=0)[::-1]
    tot = data.sum(axis=0)
    tot[tot == 0] = 1
    cum = np.cumsum(np.take_along_axis(data / tot,
                                       ss, axis=0), axis=0)
    a, b = np.where(np.diff(cum <= percentage,
                            axis=0))
    a += 1
    _ = b.argsort()
    a, b = a[_], b[_]
    values = data[ss[a, b], b]
    if values.shape[0] != data.shape[0]:
        a_n = np.zeros(data.shape[0], dtype=values.dtype)
        a_n[b] = values
        values = a_n
    data[data < values] = 0
    return data


def transform_from_tokens(enc):
    """Transform from token list"""

    def dense(text):
        token2id = enc.bow.token2id
        seq = []
        for token in text:
            try:
                seq.append(token2id[token])
            except KeyError:
                continue
        W = enc.weights
        if len(seq) == 0:
            x = np.ones(W.shape[0], dtype=W.dtype)
        else:
            x = W[:, seq].sum(axis=1)
        return x

    def inner(texts):
        X = np.r_[[dense(text) for text in texts]]
        _norm = np.linalg.norm(X, axis=1)
        _norm[_norm == 0] = 1
        return X / np.c_[_norm]

    return inner


def load_dataset(country: Union[str, list],
                 lang: str='es',
                 return_X_y:bool=False):
    """Country identification dataset"""
    if not isdir(MODELS):
        os.mkdir(MODELS)    
    if isinstance(country, str):
        country = [country]
    for cntr in country:
        url = f'{DialectID_URL}/{lang}-{cntr}-sample.json.zip'
        filename=join(MODELS, f'{lang}-{cntr}-sample.json.zip')
        if isfile(filename):
            continue
        Download(url, filename)
        with ZipFile(filename, "r") as fpt:
            fpt.extractall(path=MODELS,
                            pwd="ingeotec".encode("utf-8"))
    if len(country) == 1 and return_X_y is False:
        return join(MODELS, f'{lang}-{country[0]}-sample.json')
    assert return_X_y
    X = []
    y = []
    for cntr in country:
        _ = join(MODELS, f'{lang}-{cntr}-sample.json')
        _ = list(tweet_iterator(_))
        X.extend(_)
        y.extend([cntr] * len(_))
    return X, y