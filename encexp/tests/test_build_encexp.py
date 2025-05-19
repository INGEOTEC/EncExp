# Copyright 2025 Mario Graff (https://github.com/mgraffg)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from os.path import isfile, join
import os
import numpy as np
from sklearn.base import clone
from microtc.utils import tweet_iterator, Counter
# from encexp.tests.test_utils import samples
from encexp.utils import load_dataset
from encexp.text_repr import SeqTM, EncExpT
from encexp.build_encexp import Dataset, EncExpDataset, Train, main


def test_Dataset_output_filename():
    """Test Dataset"""
    seq = SeqTM(lang='es')
    ds = Dataset(text_model=seq)
    assert ds.output_filename == join('.', f'{seq.identifier}.tsv')


def test_Dataset_process():
    """Test Dataset process"""
    
    dataset = load_dataset('mx')
    iter = list(tweet_iterator(dataset))
    for x in iter:
        x['klass'] = 'mx'
    seq = SeqTM(lang='es', token_max_filter=2**13)
    ds = Dataset(text_model=seq, self_supervised=False)
    ds.process(iter)
    data = open(ds.output_filename, encoding='utf-8').readlines()
    assert data[0][:2] == 'mx'

    iter = list(tweet_iterator(dataset))
    seq = SeqTM(lang='es', token_max_filter=2**13)
    words = [str(x)  for x in seq.names
             if x[:2] != 'q:' and x[:2] != 'e:']
    cnt = Counter()
    cnt.update(words)
    seq.set_vocabulary(cnt)
    ds = Dataset(text_model=seq)
    ds.process(iter)
    data = open(ds.output_filename, encoding='utf-8').readlines()
    assert len(data) <= len(iter)
    assert len(data[0].split('\t')) == 2


def test_Dataset_self_supervise():
    """Test Dataset self_supervise parameter"""
    X, y = load_dataset(['mx', 'ar'], return_X_y=True)
    D = [dict(text=text, klass=klass)
         for text, klass in zip(X, y)]
    seq = SeqTM(lang='es', token_max_filter=2**13)
    ds = Dataset(text_model=seq)
    ds.process(D)
    data = open(ds.output_filename, encoding='utf-8').readlines()
    os.unlink(ds.output_filename)
    cnt = Counter()
    for x in data:
        cnt.update(x.split('\t')[0].split(" "))
    assert len(cnt) > 2


def test_EncExpDataset():
    """Test EncExpDataset"""
    
    dataset = load_dataset('mx')
    seq = SeqTM(lang='es', token_max_filter=2**13)
    ds = EncExpDataset(text_model=seq)
    iter = list(tweet_iterator(dataset))
    ds.process(iter)
    data = open(ds.output_filename, encoding='utf-8').readlines()
    assert len(data) <= len(iter)
    assert len(data[0].split('\t')) == 2


def test_Train_labels():
    """Test labels"""
    
    dataset = load_dataset('mx')
    seq = SeqTM(lang='es', token_max_filter=2**13)
    ds = EncExpDataset(text_model=clone(seq))
    ds.process(tweet_iterator(dataset))
    train = Train(text_model=seq, min_pos=32,
                  filename=ds.output_filename)
    assert len(train.labels) == 88
    X, y = load_dataset(['mx', 'ar'], return_X_y=True)
    D = [dict(text=text, klass=label) for text, label in zip(X, y)]
    ds = EncExpDataset(text_model=clone(seq), self_supervised=False)
    ds.process(D)
    train = Train(text_model=seq, min_pos=32,
                  filename=ds.output_filename)
    assert len(train.labels) == 2
    assert len(train.labels_freq) == 2
    os.unlink(ds.output_filename)


def test_Train_training_set():
    """Test Train"""
    
    dataset = load_dataset('mx')
    seq = SeqTM(lang='es', token_max_filter=2**13)
    ds = EncExpDataset(text_model=clone(seq))
    # if not isfile(ds.output_filename):
    ds.process(tweet_iterator(dataset))
    train = Train(text_model=seq, min_pos=32,
                  filename=ds.output_filename)
    labels = train.labels
    X, y = train.training_set(labels[0])
    assert X.shape[0] == len(y) and X.shape[1] == len(seq.names)
    # cnt = np.where((X > 0).sum(axis=0).A1)[0].shape
    train.keep_unfreq = True
    X, y = train.training_set(labels[0])
    _, freq =  np.unique(y, return_counts=True)
    assert freq[0] > freq[1]
    train.min_neg = 0
    X, y = train.training_set(labels[0])
    _, freq =  np.unique(y, return_counts=True)
    assert freq[0] == freq[1]

    # cnt2 = np.where((X > 0).sum(axis=0).A1)[0].shape
    # assert cnt < cnt2


def test_Train_parameters():
    """Test Train"""
    
    dataset = load_dataset('mx')
    seq = SeqTM(lang='es', token_max_filter=2**13)
    ds = EncExpDataset(text_model=clone(seq))
    if not isfile(ds.output_filename):
        ds.process(tweet_iterator(dataset))
    train = Train(text_model=seq, min_pos=32,
                  filename=ds.output_filename)
    labels = train.labels
    params = train.parameters(labels[0])
    assert 'N' in params


def test_Train_store_model():
    """Test Train"""
    
    dataset = load_dataset('mx')
    enc = EncExpT(lang='es', token_max_filter=2**13,
                  pretrained=False)
    enc.pretrained = True
    ds = EncExpDataset(text_model=clone(enc.seqTM))
    ds.identifier = enc.identifier
    if not isfile(ds.output_filename):
        ds.process(tweet_iterator(dataset))
    train = Train(text_model=enc.seqTM, min_pos=32,
                  filename=ds.output_filename)
    train.identifier = enc.identifier
    train.store_model()
    assert isfile(f'{enc.identifier}.json.gz')


def test_seqtm_build():
    """Test SeqTM CLI"""

    class A:
        """Dummy"""

    dataset = load_dataset('mx')
    A.lang = 'es'
    A.file = [dataset]
    A.voc_size_exponent = 13
    A.n_jobs = -1
    main(A)