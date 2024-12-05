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
from itertools import count
from random import randint, shuffle
import gzip
import json
import os
from os.path import isfile, basename
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
import numpy as np
from microtc.utils import tweet_iterator, Counter
import encexp
from encexp.text_repr import SeqTM
from encexp.utils import progress_bar


def encode_output(fname, prefix='encode'):
    """Encode output filename"""
    base = basename(fname)
    _ = fname.split(base)
    if isinstance(_, str):
        output = f'{prefix}-{_}'
    else:
        output = f'{_[0]}{prefix}-{base}'
    if output[-3:] == '.gz':
        return output[:-3]
    return output


def update_tokens(seq, tokens=None):
    """Update the tokens in SeqTM"""

    from microtc import emoticons
    if tokens is None:
        return seq
    for token in tokens:
        key = f'~{token}~'
        if key not in seq.tokens:
            seq.tokens[key] = token
            seq._map[key] = token
    _ = emoticons.create_data_structure
    seq.data_structure = _(seq.tokens)
    return seq


def encode(vocabulary:dict, fname: str, tokens: list=None,
           limit: int=None):
    """Encode file"""
    limit = np.inf if limit is None else limit
    loop = count() if limit == np.inf else range(limit)    
    output = encode_output(fname)
    seq = update_tokens(SeqTM(vocabulary=vocabulary),
                        tokens=tokens)
    tokenize = seq.tokenize
    cnt = Counter()
    with open(output, 'w', encoding='utf-8') as fpt:
        for tweet, _ in progress_bar(zip(tweet_iterator(fname), loop),
                                     total=limit,
                                     desc=output):
            _ = tokenize(tweet)
            cnt.update(_)
            print(json.dumps(_), file=fpt)
    return output, cnt


def feasible_tokens(vocabulary: dict, count: dict,
                    tokens: list=None,
                    min_pos: int=512):
    """Feasible tokens"""
    seq = SeqTM(vocabulary=vocabulary)
    tokens = seq.names if tokens is None else tokens
    output = []
    for k, v in enumerate(tokens):
        if count[v] < min_pos:
            continue
        output.append((k, v))
    return output


def build_encexp_token(index, vocabulary,
                       fname, max_pos=2**13,
                       precision=np.float16,
                       transform=None,
                       estimator_kwargs=None,
                       label=None):
    """Build token classifier"""
    seq = SeqTM(vocabulary=vocabulary)
    output_fname = encode_output(fname, prefix=f'{index}')
    POS = []
    NEG = []

    if isfile(output_fname):
        try:
            next(tweet_iterator(output_fname))
            return output_fname
        except Exception:
            pass
    for text in tweet_iterator(fname):
        if label in text:
            POS.append([x for x in text if x != label])
        elif len(NEG) - len(POS) < 1024:
            NEG.append(text)
        else:
            k = randint(0, len(NEG) - 1)
            del NEG[k]
        if len(POS) > max_pos:
            break
    if len(POS) == 0 or len(NEG) == 0:
        return None
    shuffle(NEG)
    NEG = NEG[:len(POS)]
    if transform is not None:
        X = transform(POS + NEG)
    else:
        X = seq.tonp([seq.model[x] for x in POS + NEG])
    y = [1] * len(POS) + [0] * len(NEG)
    est_kwargs = dict(class_weight='balanced',
                      fit_intercept=False,
                      dual='auto')
    if estimator_kwargs:
        est_kwargs.update(estimator_kwargs)
    m = LinearSVC(**est_kwargs).fit(X, y)
    coef = m.coef_[0].astype(precision)
    with open(output_fname, 'wb') as fpt:
        output = dict(N=len(y), coef=coef.tobytes().hex(),
                      intercept=float(m.intercept_), label=label)
        fpt.write(bytes(json.dumps(output), encoding='utf-8'))
    return output_fname


def build_encexp(vocabulary, fname, output,
                 min_pos: int=512, max_pos: int=2**13,
                 n_jobs: int = -1, precision=np.float16,
                 estimator_kwargs: dict=None, limit: int=None,
                 transform=None, tokens: list=None):
    """Build EncExp"""
    encode_fname, cnt = encode(vocabulary, fname, tokens=tokens, 
                               limit=limit)
    tokens = feasible_tokens(vocabulary, cnt, tokens=tokens,
                             min_pos=min_pos)
    fnames = Parallel(n_jobs=n_jobs)(delayed(build_encexp_token)(index,
                                                                 vocabulary,
                                                                 encode_fname,
                                                                 precision=precision,
                                                                 max_pos=max_pos,
                                                                 estimator_kwargs=estimator_kwargs,
                                                                 transform=transform,
                                                                 label=label)
                                     for index, label in progress_bar(tokens,
                                                                  desc=output,
                                                                  total=len(tokens)))
    with gzip.open(output, 'wb') as fpt:
        fpt.write(bytes(json.dumps(vocabulary) + '\n',
                        encoding='utf-8'))
        for fname in fnames:
            if fname is None:
                continue
            data = next(tweet_iterator(fname))
            fpt.write(bytes(json.dumps(data) + '\n',
                            encoding='utf-8'))
    for fname in fnames:
        if fname is None:
            continue        
        os.unlink(fname)
    os.unlink(encode_fname)


def main(args):
    """CLI"""
    filename  = args.file[0]
    output = args.output
    vocabulary = args.vocabulary
    n_jobs = args.n_jobs
    voc = next(tweet_iterator(vocabulary))
    min_pos = args.min_pos
    estimator_kwargs = None
    if args.intercept:
        estimator_kwargs = dict(fit_intercept=True)
    build_encexp(voc, filename, output,
                 min_pos=min_pos, limit=args.limit,
                 n_jobs=n_jobs,
                 estimator_kwargs=estimator_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute EncExp',
                                     prog='EncExp.build_encexp')
    parser.add_argument('-v', '--version', action='version',
                        version=f'EncExp {encexp.__version__}')
    parser.add_argument('-o', '--output',
                        help='Output filename',
                        dest='output', type=str)
    parser.add_argument('--vocabulary',
                        help='Vocabulary filename',
                        dest='vocabulary', type=str)
    parser.add_argument('--min-pos-examples',
                        help='Minimum number of positive examples',
                        dest='min_pos', type=int, default=512)
    parser.add_argument('--limit', help='Maximum size of the dataset',
                        dest='limit',
                        type=int, default=None)
    parser.add_argument('--intercept',
                        help='Estimate the intercept',
                        dest='intercept', action='store_true')
    parser.add_argument('--n-jobs',
                        help='Number of jobs',
                        dest='n_jobs', type=int, default=-1)    
    parser.add_argument('file',
                        help='Input filename',
                        nargs=1, type=str)
    args = parser.parse_args()
    main(args)

