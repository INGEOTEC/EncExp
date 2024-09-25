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
import os
import json
import numpy as np
from numpy.testing import assert_almost_equal
from microtc.utils import Counter, tweet_iterator
from encexp.utils import Download, DialectID_URL, compute_b4msa_vocabulary
from encexp.utils import to_float16, compute_seqtm_vocabulary, unit_length
from encexp.utils import set_to_zero


def test_download():
    """Test Download"""

    from os.path import isfile

    Download("http://github.com", "t.html")
    assert isfile("t.html")
    os.unlink("t.html")
    try:
        Download("adasdf", "t1.html")
    except ValueError:
        return
    assert False


def test_download_use_tqdm():
    """Test to disable tqdm"""

    from encexp import utils 

    utils.USE_TQDM = False
    utils.Download("http://github.com", "t.html")
    os.unlink("t.html")


def samples(filename='es-mx-sample.json.zip'):
    """Download MX sample"""

    from zipfile import ZipFile

    if isfile(filename):
        return
    Download(f'{DialectID_URL}/{filename}',
             filename)
    with ZipFile(filename, "r") as fpt:
        fpt.extractall(path=".",
                       pwd="ingeotec".encode("utf-8"))


def test_compute_b4msa_vocabulary():
    """Compute vocabulary"""

    samples()
    data = compute_b4msa_vocabulary('es-mx-sample.json')
    _ = data['counter']
    counter = Counter(_["dict"], _["update_calls"])
    assert counter.most_common()[0] == ('q:e~', 1847)
    data = compute_b4msa_vocabulary('es-mx-sample.json', 10)
    _ = data['counter']
    counter = Counter(_["dict"], _["update_calls"])
    assert counter.update_calls == 10


def test_uniform_sample():
    """Test uniform sample"""

    from encexp.utils import uniform_sample
    import numpy as np

    data = uniform_sample(10, np.array([20, 5, 4, 7]))
    assert data.sum() == 10


def test_to_float16():
    """Test to_float16"""

    fname = 't.json'
    with open(fname, 'w') as fpt:
        print(json.dumps(dict(vacio=1)), file=fpt)
        arr = np.array([1, 1.75], dtype=np.float32)
        _ = json.dumps(dict(coef=arr.tobytes().hex()))
        fpt.write(_)
    to_float16('t.json', 't2.json.gz')
    for data in tweet_iterator('t2.json.gz'):
        pass
    _ = np.frombuffer(bytearray.fromhex(data['coef']),
                      dtype=np.float16)
    assert np.all(_ == arr)
    os.unlink('t.json')
    os.unlink('t2.json.gz')


def test_compute_seqtm_vocabulary_prefix_suffix():
    """Test encode method"""
    from encexp.text_repr import SeqTM

    samples()
    data = compute_b4msa_vocabulary('es-mx-sample.json')
    voc = compute_seqtm_vocabulary(SeqTM, data,
                                   'es-mx-sample.json',
                                   voc_size_exponent=10,
                                   prefix_suffix=True)
    for k in voc['counter']['dict']:
        if k[:2] != 'q:':
            continue
        if len(k) < 6:
            assert k[3] == '~' or k[-1] == '~'


def test_unit_length():
    """Test unit length"""
    from encexp import EncExp
    enc = EncExp(lang='es',
                 precision=np.float16)
    w = enc.weights
    w_u = unit_length(w)
    a = np.sqrt((w_u * w_u).sum(axis=0))
    assert_almost_equal(a, 1, decimal=5)
    w[:, 3] = 0
    w_u = unit_length(w)
    assert w_u[:, 3].sum() == 0


def test_set_to_zero():
    """Test set to zero"""
    a = np.array([1, -1.2, 0.1])
    set_to_zero(a, percentage=1)
    assert_almost_equal(a, np.array([1, 0, 0.1]))
    a = np.array([[0.5, 0, 0.33],
                  [0.1, 0, 0.33],
                  [0.4, 0, 0.33]])
    set_to_zero(a, percentage=0.65)
    b = np.array([[0.5, 0, 0.33],
                  [0.0, 0, 0.33],
                  [0.4, 0, 0.33]])
    assert_almost_equal(a, b)

    
    