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
from microtc.utils import Counter, tweet_iterator
from encexp.tests.test_utils import samples
from encexp.utils import compute_b4msa_vocabulary, compute_seqtm_vocabulary
from encexp.text_repr import SeqTM, EncExp
from encexp.build_encexp import encode_output, encode, feasible_tokens, build_encexp_token, build_encexp
from encexp.build_voc import main, build_voc
from os.path import isfile
import os


def test_seqtm_build():
    """Test SeqTM CLI"""

    class A:
        """Dummy"""


    samples()
    A.lang = 'en'
    A.file = ['es-mx-sample.json']
    A.output = None
    A.limit = None
    A.voc_size_exponent = 4
    A.prefix_suffix = True
    main(A)
    data = next(tweet_iterator('seqtm_en_4.json.gz'))
    _ = data['counter']
    counter2 = Counter(_["dict"], _["update_calls"])
    assert counter2.most_common()[0] == ('q:a~', 1813)
    os.unlink('seqtm_en_4.json.gz')


def test_build_voc():
    """Test build voc"""
    samples()
    build_voc('es-mx-sample.json', output='t.json.gz')
    os.unlink('t.json.gz')


def test_encexp_encode():
    """Test encode method"""
    samples()
    data = compute_b4msa_vocabulary('es-mx-sample.json')
    voc = compute_seqtm_vocabulary(SeqTM, data,
                                   'es-mx-sample.json',
                                   voc_size_exponent=10)
    output, cnt = encode(voc, 'es-mx-sample.json')
    assert isfile(output)
    assert output == 'encode-es-mx-sample.json'
    os.unlink('encode-es-mx-sample.json')


def test_encexp_encode_output():
    """Test EncExp encode output filename"""
    output = encode_output('bla.json.gz')
    assert output == 'encode-bla.json'
    output = encode_output('/data/bla.json')
    assert output == '/data/encode-bla.json'


def test_feasible_tokens():
    """Test feasible tokens"""
    samples()
    data = compute_b4msa_vocabulary('es-mx-sample.json')
    voc = compute_seqtm_vocabulary(SeqTM, data,
                                   'es-mx-sample.json',
                                   voc_size_exponent=10)
    output, cnt = encode(voc, 'es-mx-sample.json')
    tokens = feasible_tokens(voc, cnt)
    assert len(tokens) == 11
    os.unlink('encode-es-mx-sample.json')


def test_build_encexp_token():
    """Test build token classifier"""
    samples()
    data = compute_b4msa_vocabulary('es-mx-sample.json')
    voc = compute_seqtm_vocabulary(SeqTM, data,
                                   'es-mx-sample.json',
                                   voc_size_exponent=10)
    output, cnt = encode(voc, 'es-mx-sample.json')
    tokens = feasible_tokens(voc, cnt)
    index, token = tokens[-3]
    fname = build_encexp_token(index, voc, output, label=token)
    assert fname == f'{index}-encode-es-mx-sample.json'
    os.unlink('encode-es-mx-sample.json')
    data = next(tweet_iterator(fname))
    assert data['label'] == token
    os.unlink(fname)


def test_build_encexp():
    """Test build encexp"""
    samples()
    data = compute_b4msa_vocabulary('es-mx-sample.json')
    voc = compute_seqtm_vocabulary(SeqTM, data,
                                   'es-mx-sample.json',
                                   voc_size_exponent=13)
    build_encexp(voc, 'es-mx-sample.json', 'encexp-es-mx.json.gz',
                 min_pos=16)
    assert isfile('encexp-es-mx.json.gz')
    lst = list(tweet_iterator('encexp-es-mx.json.gz'))
    assert lst[1]['intercept'] == 0
    os.unlink('encexp-es-mx.json.gz')
    tokens = set(SeqTM(vocabulary=voc).names)
    tokens_w = set([x['label'] for x in lst[1:]])
    assert len(tokens_w - tokens) == 0


def test_build_encexp_estimator_kwargs():
    """Test build encexp with estimator_kwargs"""
    import numpy as np
    from encexp.text_repr import EncExp
    samples()
    data = compute_b4msa_vocabulary('es-mx-sample.json')
    voc = compute_seqtm_vocabulary(SeqTM, data,
                                   'es-mx-sample.json',
                                   voc_size_exponent=10)
    build_encexp(voc, 'es-mx-sample.json', 'encexp-es-mx.json.gz',
                 estimator_kwargs=dict(fit_intercept=True))
    assert isfile('encexp-es-mx.json.gz')
    lst = list(tweet_iterator('encexp-es-mx.json.gz'))
    assert lst[1]['intercept'] != 0
    enc = EncExp(EncExp_filename='encexp-es-mx.json.gz',
                 precision=np.float16)
    assert enc.bias.shape[0] == enc.weights.shape[0]
    os.unlink('encexp-es-mx.json.gz')
    build_encexp(voc, 'es-mx-sample.json', 'encexp-es-mx.json.gz',
                 limit=10,
                 estimator_kwargs=dict(fit_intercept=True))
    assert isfile('encexp-es-mx.json.gz')
    os.unlink('encexp-es-mx.json.gz')


def test_build_encexp_transform():
    """Test build encexp EncExp.transform"""
    from encexp.download import download_encexp
    from encexp import EncExp
    import numpy as np

    samples()
    enc = EncExp(lang='es', precision=np.float16,
                 prefix_suffix=True)
    voc = download_encexp(lang='es', precision=np.float16,
                          voc_source='noGeo',
                          prefix_suffix=True)['seqtm']

    build_encexp(voc, 'es-mx-sample.json', 'encexp-es-mx.json.gz',
                 transform=enc.transform,
                 estimator_kwargs=dict(fit_intercept=True))
    assert isfile('encexp-es-mx.json.gz')
    lst = list(tweet_iterator('encexp-es-mx.json.gz'))
    assert lst[1]['intercept'] != 0
    os.unlink('encexp-es-mx.json.gz')


def test_build_encexp_tokens():
    """Test encexp with specific topics"""
    import numpy as np
    from b4msa import TextModel
    from microtc.utils import Counter
    from encexp.download import download_seqtm
    from encexp.utils import b4msa_params, replace_tokens
    

    samples()
    params = b4msa_params(lang='es')
    tokenize = replace_tokens(TextModel(**params)).tokenize    
    cnt = Counter()
    for txt in tweet_iterator('es-mx-sample.json'):
        cnt.update([x for x in tokenize(txt) if x[:2] != 'q:'])
    voc = download_seqtm(lang='es', voc_source='noGeo')
    # words = set(cnt.keys()) - set(voc['counter']['dict'])
    words = [word for word in cnt if cnt[word] >= 8]
    words.append('de')
    words = sorted(words)
    output, cnt = encode(voc, 'es-mx-sample.json', tokens=words)
    tokens = feasible_tokens(voc, cnt, tokens=words,
                             min_pos=8)
    fname = build_encexp_token(0, voc, output, precision=np.float16,
                               label=tokens[0][1])
    assert isfile(fname)
    assert next(tweet_iterator(fname))['label'] == tokens[0][1]

    # assert isfile(output)
    # assert output == 'encode-es-mx-sample.json'
    # os.unlink('encode-es-mx-sample.json')
    build_encexp(voc, 'es-mx-sample.json', 'encexp-es-mx.json.gz',
                 tokens=words, min_pos=8)
    assert isfile('encexp-es-mx.json.gz')
    enc = EncExp(lang=None, voc_source=None,
                 EncExp_filename='encexp-es-mx.json.gz',
                 precision=np.float16)
    assert enc.weights.shape[0] == len(tokens)
    os.unlink('encexp-es-mx.json.gz')
    assert np.all(enc.names == np.array(words))
    