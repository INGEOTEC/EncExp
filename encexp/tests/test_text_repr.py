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
    D = load_dataset(dataset='dev')[:2048]
    enc = EncExpT(lang='es', pretrained=False)
    enc.tailored(D, tsv_filename='tailored.tsv',
                 min_pos=32,
                 filename='tailored.json.gz')
    assert enc.weights.shape[0] == 2**14
    assert enc.weights.shape[1] == 91
    W = enc.encode('buenos dias')
    assert  W.shape == (1, 91)
    X = enc.transform(D)
    assert X.shape == (2048, 91)


def test_EncExpT_pretrained():
    """Test EncExpT pretrained"""
    enc = EncExpT(lang='es', token_max_filter=2**13)
    X = enc.transform(['buenos dias'])
    assert X.shape == (1, 4977)
    assert len(enc.names) == 4977


def test_EncExpT_tailored_intercept():
    """Test EncExpT tailored"""
    D = load_dataset(dataset='dev')[:2048]
    enc = EncExpT(lang='es', with_intercept=True,
                  pretrained=False)
    enc.tailored(D, tsv_filename='tailored.tsv',
                 min_pos=32,
                 filename='tailored_intercept.json.gz')
    assert enc.weights.shape[0] == 2**14
    assert enc.weights.shape[1] == 91
    assert enc.intercept.shape[0] == 91
    X = enc.transform(['buenos dias'])
    assert X.shape[1] == 91
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
    D = load_dataset(dataset='dev')[:2048]
    enc = EncExpT(lang='es', token_max_filter=2**13)
    enc.tailored(D, min_pos=32)


def test_EncExpT_tailored_no_neg():
    """Test EncExpT tailored"""
    dataset = load_dataset(dataset='dev')[:2048]
    D = [f'{text} de' for text in dataset]
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
    D = load_dataset(dataset='dev')[:2048]
    tm = TextModel(del_diac=False, pretrained=False).fit(D)
    cdn = normalize('NFD', 'ñ')
    lst = [x for x in tm.names if cdn in x]
    assert len(lst) > 3
    cdn = normalize('NFD', 'á')
    lst = [x for x in tm.names if cdn in x]
    assert len(lst) > 3


def test_EncExpT_transform_dtype():
    """Test EncExpT transform type"""
    enc = EncExpT(lang='es',
                  token_max_filter=2**13)
    X = enc.transform(['buenos dias'])
    assert X.dtype == enc.precision


def test_EncExpT_encode():
    """Test EncExpT transform type"""
    enc = EncExpT(lang='es', merge_encode=False,
                  token_max_filter=2**13)
    text = 'el infarto tiene que ver con el organo'
    index = [k for k, v in enumerate(enc.seqTM.tokenize(text)) if v == 'el']
    X = enc.encode(text)
    for otro in index[1:]:
        assert_almost_equal(X[index[0]], X[otro])
    enc.merge_encode = True
    X2 = enc.encode(text)
    assert_almost_equal(X.sum(axis=0), X.sum(axis=0))
    