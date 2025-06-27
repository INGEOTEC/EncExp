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
import os
import numpy as np
from numpy.testing import assert_almost_equal
from sklearn.base import clone
from microtc.utils import tweet_iterator
# from encexp.tests.test_utils import samples
from encexp.utils import load_dataset, MODEL_LANG
from encexp.text_repr import TextModel, SeqTM, EncExpT


def test_TextModel():
    """Test TextModel"""
    tm = TextModel(lang='ja', pretrained=False)
    assert tm.token_list == [1, 2, 3]
    tm = TextModel(lang=None, pretrained=False)
    assert tm.token_list == [-2, -1, 2, 3, 4]
    tm = TextModel(token_list=[2, 1], pretrained=False)
    assert tm.token_list == [2, 1]


def test_TextModel_normalize():
    """Test TextModel token normalization"""

    tm = TextModel(pretrained=False)
    txt = tm.text_transformations('e s💁🏿 🤣🤣na')
    assert txt == '~e~s~e:💁~e:🤣~e:🤣~na~'
    tm = TextModel(norm_punc=True, del_punc=False, pretrained=False)
    txt = tm.text_transformations('es💁🏿.🤣,🤣 XXX')
    assert txt == '~es~e:💁~e:.~e:🤣~e:,~e:🤣~xxx~'


def test_TextModel_tokenize():
    """Test TextModel tokenize"""
    tm = TextModel(token_list=[-1, 1], pretrained=False)
    tokens = tm.tokenize('hola💁🏿 🤣dios')
    assert tokens == ['hola', 'e:💁', 'e:🤣', 'dios', 'q:~', 'q:h',
                      'q:o', 'q:l', 'q:a', 'q:~', 'q:~', 'q:d', 
                      'q:i', 'q:o', 'q:s', 'q:~']
    tm = TextModel(token_list=[7], q_grams_words=False, pretrained=False)
    tokens = tm.tokenize('buenos💁🏿dia colegas _url _usr')
    assert tokens == ['q:~buenos', 'q:buenos~', 'q:~dia~co',
                      'q:dia~col', 'q:ia~cole', 'q:a~coleg',
                      'q:~colega', 'q:colegas', 'q:olegas~']
    

def test_TextModel_get_params():
    """Test TextModel get_params"""
    tm = TextModel(token_list=[-1, 1], pretrained=False)
    kwargs = tm.get_params()
    assert kwargs['token_list'] == [-1, 1]


def test_TextModel_identifier():
    """test TextModel identifier"""
    import hashlib

    tm = TextModel(lang='zh', pretrained=False)
    diff = tm.identifier
    cdn = ' '.join([f'{k}={v}'
                    for k, v in [('lang', 'zh'), ('pretrained', False)]])
    _ = hashlib.md5(bytes(cdn, encoding='utf-8')).hexdigest()
    assert f'TextModel_{_}' == diff
    tm = TextModel(lang='es', pretrained=False)
    diff = tm.identifier
    cdn = ' '.join([f'{k}={v}'
                    for k, v in [('lang', 'es'), ('pretrained', False)]])
    _ = hashlib.md5(bytes(cdn, encoding='utf-8')).hexdigest()
    assert f'TextModel_{_}' == diff


def test_TextModel_pretrained():
    """test TextModel pretrained"""
    tm = TextModel(lang='es')
    assert len(tm.names) == 2**17


def test_SeqTM_TM():
    """test SeqTM based on TextModel"""
    from encexp.download import download_TextModel

    seq = SeqTM(lang='es', token_max_filter=2**13,
                pretrained=False)
    tm = TextModel(lang='es')
    voc = download_TextModel(tm.identifier)['vocabulary']
    voc['dict'] = {k: v for k, v in voc['dict'].items()
                   if k[:2] == 'q:' or '~' not in k[1:-1]}
    seq.set_vocabulary(voc)
    _ = seq.tokenize('buenos dias.?, . 😂tengan')
    assert _ == ['buenos', 'dias', 'e:.', 'e:?', 'e:,', 'e:.', 'e:😂', 'tengan']
    assert seq.pretrained
    seq = SeqTM(lang='es', token_max_filter=2**13)
    _ = seq.tokenize('buenos dias .?,')
    assert _ == ['buenos~dias', 'e:.~e:?', 'e:,']


def test_SeqTM_empty_punc():
    """Test empty punc"""

    # X, y = load_dataset(['mx', 'ar'], return_X_y=True)
    seq = SeqTM(lang='es')
    # tokens = seq.tokenize(X[0])
    # assert 'q:~e' not in tokens
    # assert 'e:' in tokens
    for lang in MODEL_LANG:
        if lang in ('ja', 'zh'):
            continue
        seq = SeqTM(lang=lang)
        assert '~e:' in seq.tokens
        assert seq.token_id['~e:'] == 'e:'


def test_EncExpT_identifier():
    """Test EncExpT identifier"""
    enc = EncExpT(lang='es')
    assert enc.identifier == 'EncExpT_c69aaba0f1b0783f273f85de6f599132'
    enc = EncExpT(lang='es', use_tqdm=False,
                  pretrained=False,
                  token_max_filter=2**14)
    assert enc.identifier == 'EncExpT_c69aaba0f1b0783f273f85de6f599132'


def test_EncExpT_tailored():
    """Test EncExpT tailored"""
    dataset = load_dataset('mx')
    D = list(tweet_iterator(dataset))
    enc = EncExpT(lang='es', pretrained=False)
    enc.tailored(D, tsv_filename='tailored.tsv',
                 min_pos=32,
                 filename='tailored.json.gz')
    assert enc.weights.shape[0] == 2**14
    assert enc.weights.shape[1] == 90
    W = enc.encode('buenos dias')
    assert  W.shape == (1, 90)
    X = enc.transform(D)
    assert X.shape == (2048, 90)


def test_EncExpT_pretrained():
    """Test EncExpT pretrained"""
    enc = EncExpT(lang='es', token_max_filter=2**13)
    X = enc.transform(['buenos dias'])
    assert X.shape == (1, 4985)
    assert len(enc.names) == 4985


def test_EncExpT_tailored_intercept():
    """Test EncExpT tailored"""
    dataset = load_dataset('mx')
    D = list(tweet_iterator(dataset))
    enc = EncExpT(lang='es', with_intercept=True,
                  pretrained=False)
    enc.tailored(D, tsv_filename='tailored.tsv',
                 min_pos=32,
                 filename='tailored_intercept.json.gz')
    assert enc.weights.shape[0] == 2**14
    assert enc.weights.shape[1] == 90
    assert enc.intercept.shape[0] == 90
    X = enc.transform(['buenos dias'])
    assert X.shape[1] == 90
    enc.with_intercept = False
    assert np.fabs(X - enc.transform(['buenos dias'])).sum() != 0
    enc.with_intercept = True
    X = enc.transform(D)
    X2 = enc.seqTM.transform(D) @ enc.weights
    X2 += enc.intercept
    assert_almost_equal(X, X2, decimal=5)
    enc.merge_encode = False
    X = enc.transform(D)
    assert_almost_equal(X, X2, decimal=5)


def test_EncExpT_tailored_add():
    """Test EncExpT tailored"""
    dataset = load_dataset('mx')
    D = list(tweet_iterator(dataset))
    enc = EncExpT(lang='es', token_max_filter=2**13)
    enc.tailored(D, min_pos=32)


def test_EncExpT_tailored_no_neg():
    """Test EncExpT tailored"""
    dataset = load_dataset('mx')
    D = [f'{text} de' for text in tweet_iterator(dataset)]
    enc = EncExpT(lang='es', token_max_filter=2**13)
    enc.tailored(D, min_pos=32)


def test_EncExpT_tailored_2cl():
    """Test EncExpT tailored"""
    X, y = load_dataset(['mx', 'ar'], return_X_y=True)
    D = [dict(text=text, klass=label) for text, label in zip(X, y)]
    enc = EncExpT(lang='es', pretrained=False,
                  with_intercept=True,
                  token_max_filter=2**13)
    enc.tailored(D, self_supervised=False, min_pos=32)
    assert enc.names.tolist() == ['ar', 'mx']
    

def test_EncExpT_norm():
    """Test EncExpT norm"""
    enc = EncExpT(lang='es',
                  distance=True,
                  token_max_filter=2**13)
    assert enc.norm.shape[0] == len(enc.names)
    X1 = enc.transform(['buenos dias'])
    enc.distance = False
    X2 = enc.transform(['buenos dias'])
    assert np.fabs(X1 - X2).sum() != 0


def test_TextModel_diac():
    """Test TextModel diac"""
    from unicodedata import normalize
    dataset = load_dataset('mx')
    D = list(tweet_iterator(dataset))
    tm = TextModel(del_diac=False, pretrained=False).fit(D)
    cdn = normalize('NFD', 'ñ')
    lst = [x for x in tm.names if cdn in x]
    assert len(lst) > 3
    cdn = normalize('NFD', 'á')
    lst = [x for x in tm.names if cdn in x]
    assert len(lst) > 3



# def test_EncExp_filename():
#     """Test EncExp"""
#     if not isfile('encexp-es-mx.json.gz'):
#         samples()
#         data = compute_b4msa_vocabulary('es-mx-sample.json')
#         voc = compute_seqtm_vocabulary(SeqTM, data,
#                                        'es-mx-sample.json',
#                                        voc_size_exponent=10)
#         build_encexp(voc, 'es-mx-sample.json', 'encexp-es-mx.json.gz')
#     enc = EncExp(EncExp_filename='encexp-es-mx.json.gz')
#     assert enc.weights.dtype == np.float32
#     assert len(enc.names) == 12
#     os.unlink('encexp-es-mx.json.gz')
    

# def test_EncExp():
#     """Test EncExp"""
#     enc = EncExp(precision=np.float16)
#     assert enc.weights.dtype == np.float16
#     assert len(enc.names) == 8192


# def test_EncExp_encode():
#     """Test EncExp encode"""

#     dense = EncExp(precision=np.float16)
#     assert dense.encode('buenos días').shape[1] == 2


# def test_EncExp_transform():
#     """Test EncExp transform"""

#     encexp = EncExp()
#     X = encexp.transform(['buenos dias'])
#     assert X.shape[0] == 1
#     assert X.shape[1] == 8192
#     assert X.dtype == np.float32


# def test_EncExp_prefix_suffix():
#     """Test EncExp prefix/suffix"""

#     encexp = EncExp(lang='es',
#                     precision=np.float16,
#                     prefix_suffix=True)
#     for k in encexp.bow.names:
#         if k[:2] != 'q:':
#             continue
#         if len(k) >= 6:
#             continue
#         assert k[3] == '~' or k[-1] == '~'


# def test_EncExp_fit():
#     """Test EncExp fit"""
#     from sklearn.svm import LinearSVC
#     samples()
#     mx = list(tweet_iterator('es-mx-sample.json'))
#     samples(filename='es-ar-sample.json.zip')
#     ar = list(tweet_iterator('es-ar-sample.json'))
#     y = ['mx'] * len(mx)
#     y += ['ar'] * len(ar)
#     enc = EncExp(lang='es',
#                  prefix_suffix=True,
#                  precision=np.float16).fit(mx + ar, y)
#     assert isinstance(enc.estimator, LinearSVC)
#     hy = enc.predict(ar)
#     assert hy.shape[0] == len(ar)
#     df = enc.decision_function(ar)
#     assert df.shape[0] == len(ar)
#     assert df.dtype == np.float64


# def test_EncExp_fit_sgd():
#     """Test EncExp fit"""
#     from sklearn.linear_model import SGDClassifier
#     from itertools import repeat
#     samples()
#     mx = list(tweet_iterator('es-mx-sample.json'))
#     samples(filename='es-ar-sample.json.zip')
#     ar = list(tweet_iterator('es-ar-sample.json'))
#     y = ['mx'] * len(mx)
#     y += ['ar'] * len(ar)
#     D = mx + ar
#     # while len(D) < 2**17:
#     for i in range(5):
#         D.extend(D)
#         y.extend(y)
#     D.append(D[0])
#     y.append(y[0])
#     enc = EncExp(lang='es').fit(D, y)
#     assert isinstance(enc.estimator, SGDClassifier)
#     hy = enc.predict(ar)
#     assert hy.shape[0] == len(ar)
#     df = enc.decision_function(ar)
#     assert df.shape[0] == len(ar)
#     assert df.dtype == np.float64    


# def test_EncExp_train_predict_decision_function():
#     """Test EncExp train_predict_decision_function"""
#     samples()
#     mx = list(tweet_iterator('es-mx-sample.json'))
#     samples(filename='es-ar-sample.json.zip')
#     ar = list(tweet_iterator('es-ar-sample.json'))
#     samples(filename='es-es-sample.json.zip')
#     es = list(tweet_iterator('es-es-sample.json'))
#     y = ['mx'] * len(mx)
#     y += ['ar'] * len(ar)
#     enc = EncExp(lang='es',
#                  prefix_suffix=True,
#                  precision=np.float16)
#     hy = enc.train_predict_decision_function(mx + ar, y)
#     assert hy.ndim == 2 and hy.shape[0] == len(y) and hy.shape[1] == 1
#     y += ['es'] * len(es)
#     hy = enc.train_predict_decision_function(mx + ar + es, y)
#     assert hy.shape[1] == 3 and hy.shape[0] == len(y)


# def test_EncExp_clone():
#     """Test EncExp clone"""

#     enc = EncExp(lang='es', prefix_suffix=True,
#                  precision=np.float16)
#     enc2 = clone(enc)
#     assert isinstance(enc2, EncExp)
#     assert np.all(enc2.weights == enc.weights)


# def test_EncExp_merge_IDF():
#     """Test EncExp without keyword's weight"""

#     enc = EncExp(lang='es', prefix_suffix=True,
#                  precision=np.float16, merge_IDF=False,
#                  force_token=False)
#     enc.fill(inplace=True)
    
#     for k, v in enc.bow.token2id.items():
#         assert enc.weights[v, v] == 0
#     enc2 = EncExp(lang='es', prefix_suffix=True,
#                   precision=np.float16, merge_IDF=True,
#                   force_token=False)
#     enc2.fill(inplace=True)
#     _ = (enc.weights * enc.bow.weights).astype(enc.precision)
#     assert_almost_equal(_, enc2.weights, decimal=5)


# def test_EncExp_fill():
#     """Test EncExp fill weights"""
#     from encexp.download import download_seqtm

#     voc = download_seqtm(lang='es')
#     samples()
#     if not isfile('encexp-es-mx.json.gz'):
#         build_encexp(voc, 'es-mx-sample.json', 'encexp-es-mx.json.gz',
#                      min_pos=64)
#     enc = EncExp(EncExp_filename='encexp-es-mx.json.gz')
#     iden = {v:k for k, v in enumerate(enc.bow.names)}
#     comp = [x for x in enc.bow.names if x not in enc.names]
#     key = enc.names[0]
#     enc.weights
#     w = enc.fill()
#     assert np.any(w[iden[key]] != 0)
#     assert_almost_equal(w[iden[comp[0]]], 0)
#     os.unlink('encexp-es-mx.json.gz')
#     assert np.all(enc.names == enc.bow.names)


# def test_EncExp_iadd():
#     """Test EncExp iadd"""

#     from encexp.download import download_seqtm

#     voc = download_seqtm(lang='es')
#     samples()
#     if not isfile('encexp-es-mx.json.gz'):
#         build_encexp(voc, 'es-mx-sample.json', 'encexp-es-mx.json.gz',
#                      min_pos=64)
#     enc = EncExp(EncExp_filename='encexp-es-mx.json.gz')
#     w = enc.weights
#     enc += enc
#     assert_almost_equal(w, enc.weights, decimal=4)
#     os.unlink('encexp-es-mx.json.gz')
#     enc2 = EncExp(lang='es', voc_source='noGeo')
#     enc2 += enc
#     enc2 = EncExp(lang='es', voc_source='noGeo')
#     r = enc2 + enc2
#     r.weights[:, :] = 0
#     assert enc2.weights[0, 0] != 0


# def test_EncExp_force_tokens():
#     """Test force tokens"""

#     enc = EncExp(lang='es', prefix_suffix=True,
#                  precision=np.float16,
#                  force_token=False)
#     w = enc.weights
#     _max = w.max(axis=1)
#     rows = np.arange(len(enc.names))
#     cols = np.array([enc.bow.token2id[x] for x in enc.names])
#     assert_almost_equal(w[rows, cols], 0)
#     enc = EncExp(lang='es', prefix_suffix=True,
#                  precision=np.float16,
#                  force_token=True)
#     w[rows, cols] = _max
#     assert_almost_equal(enc.weights, w)
#     enc = EncExp(lang='es', prefix_suffix=True,
#                  precision=np.float16, merge_IDF=False,
#                  force_token=False)
#     assert enc.weights[0, 0] == 0
#     enc.force_tokens_weights(IDF=True)
#     enc2 = EncExp(lang='es', prefix_suffix=True,
#                   precision=np.float16, merge_IDF=False,
#                   force_token=True)
#     assert enc.weights[0, 0] != enc2.weights[0, 0]
#     assert_almost_equal(enc.weights[0, 1:], enc2.weights[0, 1:])


# def test_EncExp_enc_training_size():
#     """Test training size of the embeddings"""

#     enc = EncExp(lang='es')
#     assert isinstance(enc.enc_training_size, dict)
#     for k in enc.enc_training_size:
#         assert k in enc.names


# def test_EncExp_distance():
#     """Test distance to hyperplane"""

#     txt = 'buenos días'
#     enc = EncExp(lang='es', transform_distance=True)
#     assert enc.weights_norm.shape[0] == enc.weights.shape[0]
#     X = enc.transform([txt])
#     X2 = EncExp(lang='es',
#                 transform_distance=False).transform([txt])
#     assert np.fabs(X - X2).sum() != 0


# def test_EncExp_unit_vector():
#     """Test distance to hyperplane"""

#     txt = 'buenos días'
#     enc = EncExp(lang='es', unit_vector=False)
#     X = enc.transform([txt])
#     assert np.linalg.norm(X) != 1
#     enc = EncExp(lang='es')
#     X = enc.transform([txt])
#     assert_almost_equal(np.linalg.norm(X), 1)


# def test_EncExp_build_tailored():
#     """Test the development of tailored models"""

#     samples()
#     mx = list(tweet_iterator('es-mx-sample.json'))
#     samples(filename='es-ar-sample.json.zip')
#     ar = list(tweet_iterator('es-ar-sample.json'))
#     y = ['mx'] * len(mx)
#     y += ['ar'] * len(ar)

#     enc = EncExp(lang='es',
#                  tailored=True)
#     w = enc.weights
#     enc.build_tailored(mx + ar, load=True)    
#     assert isfile(enc.tailored)
#     assert hasattr(enc, '_tailored_built')
#     enc = EncExp(lang='es',
#                  tailored=enc.tailored).fit(mx + ar, y)
#     assert np.fabs(w - enc.weights).sum() != 0
#     enc2 = clone(enc)
#     assert hasattr(enc2, '_tailored_built')
#     assert hasattr(enc2, '_estimator')
#     # os.unlink(enc.tailored)


# def test_pipeline_encexp():
#     """Test Pipeline in EncExpT"""
#     from sklearn.pipeline import Pipeline
#     from sklearn.svm import LinearSVC
#     from sklearn.model_selection import GridSearchCV
#     from sklearn.model_selection import StratifiedShuffleSplit

#     samples()
#     mx = list(tweet_iterator('es-mx-sample.json'))
#     samples(filename='es-ar-sample.json.zip')
#     ar = list(tweet_iterator('es-ar-sample.json'))
#     y = ['mx'] * len(mx)
#     y += ['ar'] * len(ar)

#     pipe = Pipeline([('encexp', EncExpT(lang='es')),
#                      ('cl', LinearSVC(class_weight='balanced'))])
#     params = {'cl__C': [0.01, 0.1, 1, 10],
#               'encexp__voc_source': ['mix', 'noGeo']}
#     sss = StratifiedShuffleSplit(random_state=0,
#                                 n_splits=1,
#                                 test_size=0.3)

#     grid = GridSearchCV(pipe,
#                         param_grid=params,
#                         cv=sss,
#                         n_jobs=1,
#                         scoring='f1_macro').fit(mx + ar, y)
#     assert grid.best_score_ > 0.7
