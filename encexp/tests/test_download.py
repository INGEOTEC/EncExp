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
from EncExp.download import download_seqtm, download_encexp
from os.path import isfile
import os


def test_download_seqtm():
    """Test download seqtm vocabulary"""
    data = download_seqtm(lang='es', output='t.json.gz')
    assert isfile('t.json.gz')
    os.unlink('t.json.gz')
    assert len(data['counter']['dict']) == 2**13
    download_seqtm(lang='es')


def test_download_encexp():
    """Test download EncExp"""

    data = download_encexp(lang='es')
    dim = 2**13
    assert len(data['seqtm']['counter']['dict']) == dim
    assert len(data['coefs'])
    for coef in data['coefs']:
        assert coef['coef'].shape[0] == dim