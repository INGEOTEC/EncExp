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
from dataclasses import dataclass
from typing import Iterable
from random import randint, shuffle
import gzip
import json
import os
from os.path import isfile, join, isdir
from sklearn.base import clone
from sklearn.svm import LinearSVC
from joblib import Parallel, delayed
import numpy as np
from microtc.utils import tweet_iterator, Counter
import encexp
from encexp.text_repr import SeqTM, EncExpT
from encexp.utils import progress_bar
from encexp.download import download


@dataclass
class Dataset:
    """Dataset"""
    prefix: str=''
    dirname: str='.'
    text_model: SeqTM=None
    use_tqdm: bool=True
    self_supervised: bool=True

    @property
    def identifier(self):
        """identifier"""
        try:
            return self._identifier
        except AttributeError:
            self.identifier = self.text_model.identifier
        return self._identifier

    @identifier.setter
    def identifier(self, value):
        self._identifier = value

    @property
    def output_filename(self):
        """output filename"""
        try:
            return self._output_filename
        except AttributeError:
            _ = join(self.dirname, f'{self.prefix}{self.identifier}.tsv')
            self.output_filename = _
        return self._output_filename

    @output_filename.setter
    def output_filename(self, value):
        self._output_filename = value

    def process(self, iterator: Iterable=None):
        """process data"""
        tm = self.text_model
        with open(self.output_filename, 'w',
                  encoding='utf-8') as fpt:
            for text in progress_bar(iterator,
                                     use_tqdm=self.use_tqdm,
                                     desc=f'{self.output_filename}'):
                label = None
                if isinstance(text, dict):
                    label = text.get('klass', None)
                    text = tm.get_text(text)
                if isinstance(text, (list, tuple)):
                    text = " ".join([tm.text_transformations(x) for x in text])
                else:
                    text = tm.text_transformations(text)
                if label is None or self.self_supervised:
                    label = " ".join(sorted(set(tm.compute_tokens(text)[0])))
                if len(label) == 0:
                    continue
                print(f'{label}\t{text}', file=fpt)


@dataclass
class EncExpDataset(Dataset):
    """EncExp Dataset"""
    @property
    def keywords(self):
        """TextModel"""
        try:
            return self._keywords
        except AttributeError:
            words = download('keywords')[self.text_model.lang]
            cnt = Counter()
            cnt.update(words)
            self.keywords = cnt
        return self._keywords

    @keywords.setter
    def keywords(self, value):
        self._keywords = value

    def process(self, iterator = None):
        self.text_model.set_vocabulary(self.keywords)
        return super().process(iterator)


@dataclass
class Train:
    """Train"""
    text_model: SeqTM=None
    min_pos: int=512
    max_pos: int=int(2**15)
    min_neg: int=int(2**14)
    filename: str=None
    use_tqdm: bool=True
    with_intercept: bool=False
    n_jobs: int=-1
    self_supervised: bool=True
    keep_unfreq: bool=False

    @property
    def identifier(self):
        """identifier"""
        try:
            return self._identifier
        except AttributeError:
            self.identifier = self.text_model.identifier
        return self._identifier
    
    @identifier.setter
    def identifier(self, value):
        self._identifier = value    

    @property
    def estimator(self):
        try:
            return clone(self._estimator)
        except AttributeError:
            _ = LinearSVC(class_weight='balanced',
                          fit_intercept=self.with_intercept)
            self.estimator = _
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        self._estimator = value

    @property
    def labels(self):
        """Labels"""
        if hasattr(self, '_labels'):
            return self._labels
        cnt = Counter()
        with open(self.filename, encoding='utf-8') as fpt:
            for line in fpt:
                line = line.strip()
                labels, text = line.split('\t')
                labels = labels.split()
                cnt.update(labels)
        labels = sorted([k for k, v in cnt.items() if v >= self.min_pos])
        self.labels = labels
        self.labels_freq = cnt
        return labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    @property
    def labels_freq(self):
        """Labels frequency"""
        return self._labels_freq

    @labels_freq.setter
    def labels_freq(self, value):
        self._labels_freq = value

    def transform(self, data: list):
        """Transform"""
        model = self.text_model.model
        _ = [model[x] for x in data]
        return self.text_model.tonp(_)

    def filter_tokens(self, tokens, label):
        """Filter tokens on self-supervised learning"""
        if not self.self_supervised:
            return tokens
        return [x for x in tokens if x != label]

    def training_set(self, label):
        """Training set"""
        self.text_model.disable_text_transformations = True
        tokenize = self.text_model.tokenize
        max_pos = min(self.max_pos,
                      self.labels_freq[label])
        num_neg = max(max_pos, self.min_neg)
        POS = []
        NEG = []
        labels_freq = [(k, v) for k, v in self.labels_freq.items() if k != label]
        with open(self.filename, encoding='utf-8') as fpt:
            for line in fpt:
                if len(POS) >= max_pos and len(NEG) >= num_neg:
                    break
                line = line.strip()
                labels, text = line.split('\t')
                labels = labels.split()
                tokens = tokenize(text)
                if label in labels:
                    if len(POS) < max_pos:
                        _ = self.filter_tokens(tokens, label)
                        POS.append(_)
                    continue
                klass, _ = min(labels_freq, key=lambda x: x[1])
                neg = dict(tokens=tokens, label=klass)
                if len(NEG) < num_neg:
                    NEG.append(neg)
                    continue
                k = randint(0, len(NEG) - 1)
                if not self.keep_unfreq:
                    NEG[k] = neg
                    continue
                if self.labels_freq[NEG[k]['label']] > self.labels_freq[neg['label']]:
                    NEG[k] = neg
        if len(NEG) == 0 or len(POS) == 0:
            return None
        NEG = [x['tokens'] for x in NEG]
        X = self.transform(POS + NEG)
        y = [1] * len(POS) + [-1] * len(NEG)
        return X, np.array(y)

    def parameters(self, label):
        """Parameteres"""
        _ = self.training_set(label)
        if _ is None:
            return None
        X, y = _
        m = self.estimator.fit(X, y)
        hy = m.decision_function(X)
        mask = (np.fabs(hy) >= 1) & (np.sign(hy) == y)
        coef = m.coef_[0].astype(np.float16)
        output = dict(N=len(y), coef=coef.tobytes().hex(),
                      label=label, no_sv=int(mask.sum()))
        if self.with_intercept:
            output['intercept'] = m.intercept_.astype(np.float16).tobytes().hex()
        return output
    
    def create_model(self):
        """Create model"""
        def inner(fname, label):
            if isfile(fname):
                return (fname, label)
            coef = self.parameters(label)
            if coef is None:
                return None
            with open(fname, 'w', encoding='utf-8') as fpt:
                print(json.dumps(coef), file=fpt)
            return (fname, label)
        if not isdir(self.identifier):
            os.mkdir(self.identifier)
        args = [(join(self.identifier, f'{iden}.json'), label)
                for iden, label in enumerate(self.labels)]
        _ = progress_bar(args, use_tqdm=self.use_tqdm)
        args = Parallel(n_jobs=self.n_jobs)(delayed(inner)(fname, label)
                                            for fname, label in _)
        return [x for x in args if x is not None]

    def store_model(self):
        """Create and store model"""
        args = self.create_model()
        with gzip.open(f'{self.identifier}.json.gz', 'wb') as fpt:
            for fname, _ in args:
                data = next(tweet_iterator(fname))
                fpt.write(bytes(json.dumps(data) + '\n',
                          encoding='utf-8'))
        self.delete_tmps(args)

    def delete_tmps(self, args):
        """Delete the auxiliary files"""
        for fname, _ in args:
            os.unlink(fname)
        os.rmdir(self.identifier)


def main(args):
    """CLI"""
    filename  = args.file[0]
    lang = args.lang
    token_max_filter = 2**args.voc_size_exponent
    n_jobs = args.n_jobs
    enc = EncExpT(lang=lang, pretrained=False,
                  token_max_filter=token_max_filter)
    enc.pretrained = True
    ds = EncExpDataset(text_model=clone(enc.seqTM))
    ds.identifier = enc.identifier
    if not isfile(ds.output_filename):
        ds.process(tweet_iterator(filename))
    train = Train(text_model=enc.seqTM,
                  filename=ds.output_filename, n_jobs=n_jobs)
    train.identifier = enc.identifier
    train.store_model()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EncExp')
    parser.add_argument('-v', '--version', action='version',
                        version=f'EncExp {encexp.__version__}')
    parser.add_argument('--lang', help='Language (ar | ca | de | en | es | fr | hi | in | it | ko | nl | pl | pt | ru | tl | tr )', type=str, default=None)
    parser.add_argument('--voc_size_exponent',
                        help='Vocabulary size express as log2',
                        dest='voc_size_exponent',
                        type=int, default=13)
    parser.add_argument('--n-jobs',
                        help='Number of jobs',
                        dest='n_jobs', type=int,
                        default=-1)
    parser.add_argument('file',
                        help='Input filename',
                        nargs=1, type=str)
    args = parser.parse_args()
    main(args)


# from encexp.utils import MODEL_LANG
# from encexp import SeqTM
# import json

# D = {}
# for lang in MODEL_LANG:
#     seq = SeqTM(lang=lang,
#                 token_max_filter=2**13)
#     words = [str(x) for x in seq.names
#              if x[:2] != 'q:']
#     qgrams = [str(x) for x in seq.names
#               if x[:2] == 'q:' and x[2] != '~' and x[-1] != '~']
#     qgrams.sort(key=lambda x: len(x), reverse=True)
#     cnt = 4978 - len(words)
#     if cnt > 0:
#         words.extend(qgrams[:cnt])
#     words = sorted(words)
#     D[lang] = words

# with open('keywords.json', 'w', encoding='utf-8') as fpt:
#     print(json.dumps(D), file=fpt)
