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
from encexp.text_repr import SeqTM
from encexp.build_encexp import encode_output, encode, feasible_tokens, build_encexp_token, build_encexp
from os.path import isfile
import os


def test_seqtm_build():
    """Test SeqTM CLI"""

    class A:
        """Dummy"""

    from EncExp.build_voc import main
    samples()
    A.lang = 'en'
    A.file = ['es-mx-sample.json']
    A.output = None
    A.limit = -1
    A.voc_size_exponent = 4
    main(A)
    data = next(tweet_iterator('seqtm_en_4.json.gz'))
    _ = data['counter']
    counter2 = Counter(_["dict"], _["update_calls"])
    assert counter2.most_common()[0] == ('q:o~', 1776)
    os.unlink('seqtm_en_4.json.gz')


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
    fname = build_encexp_token(index, voc, output)
    assert fname == '541-encode-es-mx-sample.json'
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
                                   voc_size_exponent=10)
    build_encexp(voc, 'es-mx-sample.json', 'encexp-es-mx.json.gz')
    assert isfile('encexp-es-mx.json.gz')
    os.unlink('encexp-es-mx.json.gz')