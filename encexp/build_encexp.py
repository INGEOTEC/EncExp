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
from sklearn.svm import LinearSVC
from random import randint, shuffle
from joblib import Parallel, delayed
from os.path import isfile, basename
import numpy as np
import gzip
import json
import os
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


def encode(vocabulary, fname):
    """Encode file"""
    output = encode_output(fname)
    seq = SeqTM(vocabulary=vocabulary)
    tokenize = seq.tokenize
    cnt = Counter()
    with open(output, 'w', encoding='utf-8') as fpt:
        for tweet in progress_bar(tweet_iterator(fname),
                                  desc=output):
            _ = tokenize(tweet)
            cnt.update(_)
            print(json.dumps(_), file=fpt)
    return output, cnt


def feasible_tokens(vocabulary, count, min_pos=512):
    """Feasible tokens"""
    seq = SeqTM(vocabulary=vocabulary)
    tokens = sorted(seq.model.word2id)
    output = []
    for k, v in enumerate(tokens):
        if count[v] < min_pos:
            continue
        output.append((k, v))
    return output


def build_encexp_token(index, vocabulary,
                       fname, max_pos=2**13,
                       precision=np.float32,
                       transform=None,
                       estimator_kwargs=None):
    """Build token classifier"""
    seq = SeqTM(vocabulary=vocabulary)
    tokens = sorted(seq.model.word2id)
    label = tokens[index]
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


def build_encexp(vocabulary,
                 fname, output,
                 min_pos=512,
                 max_pos=2**13,
                 n_jobs = -1,
                 precision=np.float32,
                 estimator_kwargs=None,
                 transform=None):
    """Build EncExp"""
    encode_fname, cnt = encode(vocabulary, fname)
    tokens = feasible_tokens(vocabulary, cnt, min_pos=min_pos)
    fnames = Parallel(n_jobs=n_jobs)(delayed(build_encexp_token)(index,
                                                                 vocabulary,
                                                                 encode_fname,
                                                                 precision=precision,
                                                                 max_pos=max_pos,
                                                                 estimator_kwargs=estimator_kwargs,
                                                                 transform=transform)
                                     for index, _ in progress_bar(tokens,
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
    voc = next(tweet_iterator(vocabulary))
    min_pos = args.min_pos
    build_encexp(voc, filename, output,
                 min_pos=min_pos)


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
    parser.add_argument('file',
                        help='Input filename',
                        nargs=1, type=str)
    args = parser.parse_args()
    main(args)

