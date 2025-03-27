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

import argparse
import json
import gzip
from itertools import count
import numpy as np
from microtc.utils import tweet_iterator, Counter
from encexp.text_repr import SeqTM, TextModel
from encexp.utils import progress_bar
import encexp


def compute_vocabulary(tm, iterator):
    """Compute the vocabulary"""
    tokenize = tm.tokenize
    counter = Counter()
    for tweet in iterator:
        counter.update(set(tokenize(tweet)))
    _ = dict(update_calls=counter.update_calls,
             dict=dict(counter.most_common()))
    return dict(counter=_, params=tm.get_params())


def compute_b4msa_vocabulary(filename: str,
                             limit: int=None,
                             **kwargs):
    """Compute the vocabulary"""

    if limit is None:
        limit = np.inf
    if limit == np.inf:
        loop = count()
    else:
        loop = range(limit)

    def iterator():
        for tweet, _ in progress_bar(zip(tweet_iterator(filename),
                                         loop),
                                     total=limit,
                                     desc=filename):
            yield tweet
    tm = TextModel(**kwargs)
    return compute_vocabulary(tm, iterator())


def compute_seqtm_vocabulary(instance, vocabulary,
                             filename, limit=None,
                             voc_size_exponent=13,
                             prefix_suffix=False):
    """Compute SeqTM"""

    def current_lost_words():
        words = [w for w, _ in base_voc.most_common() if w[:2] != 'q:']
        current = words[:2**voc_size_exponent]
        lost =  words[2**voc_size_exponent:]
        return current, lost

    def tokenizer(length, current):
        length += 2
        cnt = Counter()
        for k, v in base_voc.items():
            if k[:2] != 'q:' or len(k) != length:
                continue
            if prefix_suffix and length < 6 and k[3] != '~' and k[-1] != '~':
                continue
            cnt[k] = v
        for token in current:
            freq = base_voc[token]
            if freq == 0:
                continue
            cnt[token] = freq
        cnt.update_calls = base_voc.update_calls
        _ = dict(params=vocabulary['params'], counter=cnt)
        return instance(vocabulary=_).tokenize

    def optimize_vocabulary():
        words = [token for token in base_voc if token[:2] != 'q:']
        current, _ = current_lost_words()
        lengths = sorted([length
                          for length in vocabulary['params']['token_list']
                          if length > 0], reverse=True)
        for length in progress_bar(lengths, desc='qgrams'):
            tokenize = tokenizer(length, current)
            cnt = Counter()
            vacia = set(['~'])
            for word in words:
                tokens = set(tokenize(word)) - vacia
                _ = {token: base_voc[word] for token in tokens}
                cnt.update(_)
            current = [k for k, v in cnt.most_common(n=2**voc_size_exponent)]
        return cnt.most_common(n=2**voc_size_exponent)

    limit = np.inf if limit is None else limit
    loop = count() if limit == np.inf else range(limit)
    base_voc = Counter(vocabulary['counter']["dict"],
                       vocabulary['counter']["update_calls"])
    voc = optimize_vocabulary()
    cnt = Counter(dict(voc),
                  update_calls=base_voc.update_calls)
    _ = dict(params=vocabulary['params'], counter=cnt)
    tokenize = instance(vocabulary=_).tokenize
    counter = Counter()
    for tweet, _ in progress_bar(zip(tweet_iterator(filename),
                                        loop), total=limit,
                                        desc=filename):
        counter.update(set(tokenize(tweet)))
    _ = dict(update_calls=counter.update_calls,
             dict=dict(counter.most_common()[:2**voc_size_exponent]))
    data = dict(counter=_, params=vocabulary['params'])
    return data


def build_voc(filename, lang='es',
              voc_size_exponent=13,
              limit=None, output=None,
              **kwargs):
    """Build vocabulary"""
    data = compute_b4msa_vocabulary(filename, limit=limit, lang=lang)
    voc = compute_seqtm_vocabulary(SeqTM, data, filename, limit=limit,
                                   voc_size_exponent=voc_size_exponent,
                                   **kwargs)
    seqtm = SeqTM(lang=lang, voc_size_exponent=voc_size_exponent,
                  vocabulary=voc)
    output_filename = output
    if output_filename is None:
        output_filename = seqtm.identifier + '.json.gz'
    with gzip.open(output_filename, 'wb') as fpt:
        fpt.write(bytes(json.dumps(voc), encoding='utf-8'))
    

def main(args):
    """CLI"""

    filename  = args.file[0]
    build_voc(filename, lang=args.lang,
              voc_size_exponent=args.voc_size_exponent,
              limit=args.limit, output=args.output,
              prefix_suffix=args.prefix_suffix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute SeqTM Vocabulary',
                                     prog='EncExp.build_voc')
    parser.add_argument('-v', '--version', action='version',
                        version=f'EncExp {encexp.__version__}')
    parser.add_argument('-o', '--output',
                        help='Output filename',
                        dest='output', default=None, type=str)
    parser.add_argument('--lang', help='Language (ar | ca | de | en | es | fr | hi | in | it | ko | nl | pl | pt | ru | tl | tr )',
                        type=str, default='es')
    parser.add_argument('--limit', help='Maximum size of the dataset',
                        dest='limit',
                        type=int, default=None)
    parser.add_argument('--voc_size_exponent',
                        help='Vocabulary size express as log2',
                        dest='voc_size_exponent',
                        type=int, default=13)
    parser.add_argument('--prefix-suffix', 
                        help='Restric to use prefix and suffix',
                        dest='prefix_suffix', action='store_true')
    parser.add_argument('file',
                        help='Input filename',
                        nargs=1, type=str)
    main(parser.parse_args())