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
from itertools import count
from urllib import request
from urllib.error import HTTPError
try:
    USE_TQDM = True
    from tqdm import tqdm
except ImportError:
    USE_TQDM = False
from microtc.utils import tweet_iterator, Counter
from b4msa import TextModel
import numpy as np


DialectID_URL = 'https://github.com/INGEOTEC/dialectid/releases/download/data'

class Download(object):
    """Download
    
    >>> from EncExp.utils import Download
    >>> d = Download("http://github.com", "t.html")
    """

    def __init__(self, url, output='t.tmp') -> None:
        self._url = url
        self._output = output
        try:
            request.urlretrieve(url, output, reporthook=self.progress)
        except HTTPError as exc:
            self.close()
            raise exc
        self.close()

    @property
    def tqdm(self):
        """tqdm"""

        if not USE_TQDM:
            return None
        try:
            return self._tqdm
        except AttributeError:
            self._tqdm = tqdm(total=self._nblocks, leave=False)
        return self._tqdm

    def close(self):
        """Close tqdm if used"""
        if USE_TQDM:
            self.tqdm.close()

    def update(self):
        """Update tqdm if used"""
        if USE_TQDM:
            self.tqdm.update()

    def progress(self, nblocks, block_size, total):
        """tqdm progress"""

        self._nblocks = total // block_size
        self.update()


def b4msa_params(lang='es'):
    """B4MSA default parameters"""

    from microtc.params import OPTION_DELETE, OPTION_NONE
    tm_kwargs=dict(num_option=OPTION_NONE,
                   usr_option=OPTION_DELETE,
                   url_option=OPTION_DELETE,
                   emo_option=OPTION_NONE,
                   hashtag_option=OPTION_NONE,
                   ent_option=OPTION_NONE,
                   lc=True,
                   del_dup=False,
                   del_punc=True,
                   del_diac=True,
                   select_ent=False,
                   select_suff=False,
                   select_conn=False,
                   max_dimension=False,
                   unit_vector=True,
                   q_grams_words=True)
    if lang == 'ja' or lang == 'zh':
        tm_kwargs['token_list'] = [1, 2, 3]
    else:
        tm_kwargs['token_list'] = [-1, 2, 3, 4, 5, 6]
    return tm_kwargs

def progress_bar(data, total=np.inf, **kwargs):
    """Progress bar"""

    if not USE_TQDM:
        return data
    if total == np.inf:
        total = None
    return tqdm(data, total=total, **kwargs)


def compute_vocabulary(filenames, limits=None, lang='es',
                       tokenize=None, get_text = lambda x: x['text'],
                       voc_size_exponent=-1,
                       params = None, **kwargs):
    """Compute the vocabulary"""

    if params is None:
        params = b4msa_params(lang=lang)
    params.update(kwargs)
    if tokenize is None:
        tokenize = TextModel(**params).tokenize
    if isinstance(filenames, str):
        filenames = [filenames]
    if limits is None:
        limits = np.inf
    if not isinstance(limits, list):
        limits = [limits] * len(filenames)
    counter = Counter()
    for filename, limit in zip(filenames, limits):
        if limit == np.inf:
            loop = count()
        else:
            loop = range(limit)
        for tweet, _ in progress_bar(zip(tweet_iterator(filename),
                                         loop), total=limit,
                                         desc=filename):
            counter.update(set(tokenize(get_text(tweet))))
    if voc_size_exponent > 0:
        voc = counter.most_common()[:2**voc_size_exponent]
    else:
        voc = counter.most_common()
    _ = dict(update_calls=counter.update_calls,
             dict=dict(voc))
    data = dict(counter=_, params=params)
    return data


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