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
from encexp.tests.test_utils import samples
from encexp.utils import compute_b4msa_vocabulary, compute_seqtm_vocabulary, to_float16
from encexp.build_encexp import build_encexp
from encexp.text_repr import SeqTM, EncExp
from os.path import isfile
import numpy as np
import os


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
    assert len([k for k in _ if k[:2] == 'q:']) == 30


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
    to_float16('encexp-es-mx.json.gz', 'encexp-float16-es-mx.json.gz')
    enc2 = EncExp(EncExp_filename='encexp-float16-es-mx.json.gz', 
                  precision=np.float16)
    assert enc2.weights.dtype == np.float16
    w = enc.weights
    assert np.all(enc2.weights.shape == enc.weights.shape)
    os.unlink('encexp-es-mx.json.gz')
    os.unlink('encexp-float16-es-mx.json.gz')
    

def test_EncExp():
    """Test EncExp"""
    enc = EncExp()
    assert enc.weights.dtype == np.float32
    assert len(enc.names) == 2**13


def test_EncExp_encode():
    """Test EncExp encode"""

    dense = EncExp()
    assert dense.encode('buenos días').shape[1] == 2


def test_EncExp_transform():
    """Test EncExp transform"""

    encexp = EncExp()
    X = encexp.transform(['buenos dias'])
    assert X.shape[0] == 1
    assert X.shape[1] == 2**13
    assert X.dtype == np.float32
