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
from microtc.utils import Counter
from EncExp.tests.test_utils import samples
from EncExp.utils import compute_b4msa_vocabulary
from EncExp.text_repr import SeqTM
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

    from EncExp.utils import compute_seqtm_vocabulary
    from EncExp.text_repr import SeqTM
    samples()
    data = compute_b4msa_vocabulary('es-mx-sample.json')
    voc = compute_seqtm_vocabulary(SeqTM, data,
                                   'es-mx-sample.json',
                                   voc_size_exponent=5)
    assert len(voc['counter']['dict']) == 32
    _ = voc['counter']['dict']
    assert len([k for k in _ if k[:2] == 'q:']) == 9


def test_seqtm_identifier():
    """Test SeqTM identifier"""

    samples()
    data = compute_b4msa_vocabulary('es-mx-sample.json')
    seqtm = SeqTM(vocabulary=data, lang='en', voc_size_exponent=13)
    assert seqtm.identifier == 'seqtm_en_13'


def test_seqtm_build():
    """Test SeqTM CLI"""

    from microtc.utils import tweet_iterator
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