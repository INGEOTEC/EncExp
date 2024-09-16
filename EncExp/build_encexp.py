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
from EncExp.text_repr import SeqTM
from microtc.utils import tweet_iterator, Counter
from sklearn.svm import LinearSVC
from random import randint, shuffle
from joblib import Parallel, delayed
from os.path import isdir, isfile, basename
from glob import glob
import numpy as np
import gzip
import json
import os


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
        for tweet in tweet_iterator(fname):
            _ = tokenize(tweet)
            cnt.update(_)
            print(json.dumps(_), file=fpt)
    return output, cnt


def feasible_tokens(vocabulary, count, min_examples=512):
    """Feasible tokens"""
    seq = SeqTM(vocabulary=vocabulary)
    tokens = sorted(seq.model.word2id)
    output = []
    for k, v in enumerate(tokens):
        if count[v] < min_examples:
            continue
        output.append((k, v))
    return output


def build_encexp_token(index, vocabulary,
                       fname, max_pos=2**13,
                       precision=np.float32):
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
    shuffle(NEG)
    NEG = NEG[:len(POS)]
    X = seq.tonp([seq.model[x] for x in POS + NEG])
    y = [1] * len(POS) + [0] * len(NEG)

    m = LinearSVC(class_weight='balanced', fit_intercept=False).fit(X, y)
    coef = m.coef_[0].astype(precision)
    with open(output_fname, 'wb') as fpt:
        output = dict(N=len(y), coef=coef.tobytes().hex(),
                      intercept=m.intercept_, label=label)
        fpt.write(bytes(json.dumps(output), encoding='utf-8'))
    return output_fname


# for lang in ['ar', 'fr', 'pt', 'ru']:
#     print('*'*10, f'Haciendo {lang}', '*' * 10)
#     Parallel(n_jobs=-1)(delayed(create_model)(index, lang)
#                         for index in tqdm(range(2**13)))
