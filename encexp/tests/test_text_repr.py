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
from os.path import isfile
import numpy as np
from numpy.testing import assert_almost_equal
import os
from microtc.utils import tweet_iterator
from encexp.tests.test_utils import samples
from encexp.utils import compute_b4msa_vocabulary, compute_seqtm_vocabulary
from encexp.build_encexp import build_encexp
from encexp.text_repr import SeqTM, EncExp
from sklearn.base import clone


def test_seqtm():
    """Test SeqTM"""
    
    samples()
    data = compute_b4msa_vocabulary('es-mx-sample.json')
    seqtm = SeqTM(vocabulary=data)
    _ = seqtm.tokenize('buenos dias mxeico')
    assert _ == ['buenos', 'dias', 'q:~mx', 'q:ei', 'q:co~']


def test_seqtm_vocabulary():
    """Test SeqTM vocabulary"""

    samples()
    data = compute_b4msa_vocabulary('es-mx-sample.json')
    voc = compute_seqtm_vocabulary(SeqTM, data,
                                   'es-mx-sample.json',
                                   voc_size_exponent=5)
    assert len(voc['counter']['dict']) == 32
    _ = voc['counter']['dict']
    assert len([k for k in _ if k[:2] == 'q:']) == 29


# def test_seqtm_ix_15():
#     """Test SeqTM"""
#     seqtm = SeqTM(lang='es', voc_size_exponent=15,
#                   prefix_suffix=True)
#     tokens = seqtm.tokenize('buenos dias')
#     assert tokens == ['buenos', 'dias']


def test_seqtm_identifier():
    """Test SeqTM identifier"""

    samples()
    data = compute_b4msa_vocabulary('es-mx-sample.json')
    seqtm = SeqTM(vocabulary=data, lang='en', voc_size_exponent=13)
    assert seqtm.identifier == 'seqtm_en_13'


def test_seqtm_download():
    """Test SeqTM download"""
    seqtm = SeqTM(lang='es', voc_size_exponent=13)
    cdn = seqtm.tokenize('buenos dias méxico')
    assert cdn == ['buenos', 'dias', 'mexico']


def test_EncExp_filename():
    """Test EncExp"""
    if not isfile('encexp-es-mx.json.gz'):
        samples()
        data = compute_b4msa_vocabulary('es-mx-sample.json')
        voc = compute_seqtm_vocabulary(SeqTM, data,
                                       'es-mx-sample.json',
                                       voc_size_exponent=10)
        build_encexp(voc, 'es-mx-sample.json', 'encexp-es-mx.json.gz')
    enc = EncExp(EncExp_filename='encexp-es-mx.json.gz')
    assert enc.weights.dtype == np.float32
    assert len(enc.names) == 11
    os.unlink('encexp-es-mx.json.gz')
    

def test_EncExp():
    """Test EncExp"""
    enc = EncExp(precision=np.float16)
    assert enc.weights.dtype == np.float16
    assert len(enc.names) == 8192


def test_EncExp_encode():
    """Test EncExp encode"""

    dense = EncExp(precision=np.float16)
    assert dense.encode('buenos días').shape[1] == 2


def test_EncExp_transform():
    """Test EncExp transform"""

    encexp = EncExp()
    X = encexp.transform(['buenos dias'])
    assert X.shape[0] == 1
    assert X.shape[1] == 8192
    assert X.dtype == np.float32


# def test_EncExp_transform_float16():
#     """Test EncExp transform (float16)"""

#     encexp = EncExp(voc_source='mix', prefix_suffix=False,
#                     precision=np.float16)
#     X = encexp.transform(['buenos dias'])
#     assert X.shape[0] == 1
#     assert X.shape[1] == 8132
#     assert X.dtype == np.float32


def test_EncExp_prefix_suffix():
    """Test EncExp prefix/suffix"""

    encexp = EncExp(lang='es',
                    precision=np.float16,
                    prefix_suffix=True)
    for k in encexp.bow.names:
        if k[:2] != 'q:':
            continue
        if len(k) >= 6:
            continue
        assert k[3] == '~' or k[-1] == '~'


def test_EncExp_fit():
    """Test EncExp fit"""
    from sklearn.svm import LinearSVC
    samples()
    mx = list(tweet_iterator('es-mx-sample.json'))
    samples(filename='es-ar-sample.json.zip')
    ar = list(tweet_iterator('es-ar-sample.json'))
    y = ['mx'] * len(mx)
    y += ['ar'] * len(ar)
    enc = EncExp(lang='es',
                 prefix_suffix=True,
                 precision=np.float16).fit(mx + ar, y)
    assert isinstance(enc.estimator, LinearSVC)
    hy = enc.predict(ar)
    assert hy.shape[0] == len(ar)
    df = enc.decision_function(ar)
    assert df.shape[0] == len(ar)
    assert df.dtype == np.float64


def test_EncExp_fit_sgd():
    """Test EncExp fit"""
    from sklearn.linear_model import SGDClassifier
    from itertools import repeat
    samples()
    mx = list(tweet_iterator('es-mx-sample.json'))
    samples(filename='es-ar-sample.json.zip')
    ar = list(tweet_iterator('es-ar-sample.json'))
    y = ['mx'] * len(mx)
    y += ['ar'] * len(ar)
    D = mx + ar
    # while len(D) < 2**17:
    for i in range(5):
        D.extend(D)
        y.extend(y)
    D.append(D[0])
    y.append(y[0])
    enc = EncExp(lang='es').fit(D, y)
    assert isinstance(enc.estimator, SGDClassifier)
    hy = enc.predict(ar)
    assert hy.shape[0] == len(ar)
    df = enc.decision_function(ar)
    assert df.shape[0] == len(ar)
    assert df.dtype == np.float64    


def test_EncExp_train_predict_decision_function():
    """Test EncExp train_predict_decision_function"""
    samples()
    mx = list(tweet_iterator('es-mx-sample.json'))
    samples(filename='es-ar-sample.json.zip')
    ar = list(tweet_iterator('es-ar-sample.json'))
    samples(filename='es-es-sample.json.zip')
    es = list(tweet_iterator('es-es-sample.json'))
    y = ['mx'] * len(mx)
    y += ['ar'] * len(ar)
    enc = EncExp(lang='es',
                 prefix_suffix=True,
                 precision=np.float16)
    hy = enc.train_predict_decision_function(mx + ar, y)
    assert hy.ndim == 2 and hy.shape[0] == len(y) and hy.shape[1] == 1
    y += ['es'] * len(es)
    hy = enc.train_predict_decision_function(mx + ar + es, y)
    assert hy.shape[1] == 3 and hy.shape[0] == len(y)


def test_EncExp_clone():
    """Test EncExp clone"""

    enc = EncExp(lang='es', prefix_suffix=True,
                 precision=np.float16)
    enc2 = clone(enc)
    assert isinstance(enc2, EncExp)
    assert np.all(enc2.weights == enc.weights)


def test_EncExp_merge_IDF():
    """Test EncExp without keyword's weight"""

    enc = EncExp(lang='es', prefix_suffix=True,
                 precision=np.float16, merge_IDF=False,
                 force_token=False)
    enc.fill(inplace=True)
    
    for k, v in enc.bow.token2id.items():
        assert enc.weights[v, v] == 0
    enc2 = EncExp(lang='es', prefix_suffix=True,
                  precision=np.float16, merge_IDF=True,
                  force_token=False)
    enc2.fill(inplace=True)
    _ = (enc.weights * enc.bow.weights).astype(enc.precision)
    assert_almost_equal(_, enc2.weights, decimal=5)


def test_EncExp_fill():
    """Test EncExp fill weights"""
    from encexp.download import download_seqtm

    voc = download_seqtm(lang='es')
    samples()
    if not isfile('encexp-es-mx.json.gz'):
        build_encexp(voc, 'es-mx-sample.json', 'encexp-es-mx.json.gz',
                     min_pos=64)
    enc = EncExp(EncExp_filename='encexp-es-mx.json.gz')
    iden = {v:k for k, v in enumerate(enc.bow.names)}
    comp = [x for x in enc.bow.names if x not in enc.names]
    key = enc.names[0]
    enc.weights
    w = enc.fill()
    assert np.any(w[iden[key]] != 0)
    assert_almost_equal(w[iden[comp[0]]], 0)
    os.unlink('encexp-es-mx.json.gz')
    assert np.all(enc.names == enc.bow.names)


def test_EncExp_iadd():
    """Test EncExp iadd"""

    from encexp.download import download_seqtm

    voc = download_seqtm(lang='es')
    samples()
    if not isfile('encexp-es-mx.json.gz'):
        build_encexp(voc, 'es-mx-sample.json', 'encexp-es-mx.json.gz',
                     min_pos=64)
    enc = EncExp(EncExp_filename='encexp-es-mx.json.gz')
    w = enc.weights
    enc += enc
    assert_almost_equal(w, enc.weights, decimal=4)
    os.unlink('encexp-es-mx.json.gz')
    enc2 = EncExp(lang='es', voc_source='noGeo')
    enc2 += enc
    enc2 = EncExp(lang='es', voc_source='noGeo')
    r = enc2 + enc2
    r.weights[:, :] = 0
    assert enc2.weights[0, 0] != 0


def test_EncExp_force_tokens():
    """Test force tokens"""

    enc = EncExp(lang='es', prefix_suffix=True,
                 precision=np.float16,
                 force_token=False)
    w = enc.weights
    _max = w.max(axis=1)
    rows = np.arange(len(enc.names))
    cols = np.array([enc.bow.token2id[x] for x in enc.names])
    assert_almost_equal(w[rows, cols], 0)
    enc = EncExp(lang='es', prefix_suffix=True,
                 precision=np.float16,
                 force_token=True)
    w[rows, cols] = _max
    assert_almost_equal(enc.weights, w)
    enc = EncExp(lang='es', prefix_suffix=True,
                 precision=np.float16, merge_IDF=False,
                 force_token=False)
    assert enc.weights[0, 0] == 0
    enc.force_tokens_weights(IDF=True)
    enc2 = EncExp(lang='es', prefix_suffix=True,
                  precision=np.float16, merge_IDF=False,
                  force_token=True)
    assert enc.weights[0, 0] != enc2.weights[0, 0]
    assert_almost_equal(enc.weights[0, 1:], enc2.weights[0, 1:])


# def test_EncExp_intercept():
#     """Test EncExp with intercept"""

#     enc = EncExp(lang='es', intercept=True,
#                  merge_IDF=False,
#                  force_token=True)
#     assert np.all(enc.bias != 0)


def test_SeqTM_text_transformations():
    """Test SeqTM Text Transformations"""
    seq = SeqTM()
    seq.norm_emojis = True
    assert seq.tokenize('🤣🤣') == ['🤣', '🤣']


def test_SeqTM_jaja():
    """Test SeqTM jaja"""

    seq = SeqTM()
    seq.norm_emojis = True
    txt = seq.text_transformations('jajaja')
    assert txt == '~ja~'
    txt = seq.text_transformations('hola ja')
    assert txt == '~hola~ja~'
    txt = seq.text_transformations('jajaja 🤣')
    assert txt == '~ja~🤣~'
    txt = seq.text_transformations('🧑‍')
    assert txt == '~🧑~'


def test_EncExp_enc_training_size():
    """Test training size of the embeddings"""

    enc = EncExp(lang='es')
    assert isinstance(enc.enc_training_size, dict)
    for k in enc.enc_training_size:
        assert k in enc.names


def test_EncExp_distance():
    """Test distance to hyperplane"""

    txt = 'buenos días'
    enc = EncExp(lang='es', transform_distance=True)
    assert enc.weights_norm.shape[0] == enc.weights.shape[0]
    X = enc.transform([txt])
    X2 = EncExp(lang='es',
                transform_distance=False).transform([txt])
    assert np.fabs(X - X2).sum() != 0


def test_EncExp_unit_vector():
    """Test distance to hyperplane"""

    txt = 'buenos días'
    enc = EncExp(lang='es', unit_vector=False)
    X = enc.transform([txt])
    assert np.linalg.norm(X) != 1
    enc = EncExp(lang='es')
    X = enc.transform([txt])
    assert_almost_equal(np.linalg.norm(X), 1)
