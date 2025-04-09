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
from microtc.utils import Counter, tweet_iterator
from encexp.tests.test_utils import samples
from encexp.text_repr import SeqTM, EncExp
from encexp.build_voc import compute_TextModel_vocabulary, compute_SeqTM_vocabulary


def test_compute_TextModel_vocabulary():
    """Compute vocabulary"""
    def iterator():
        """iterator"""
        return tweet_iterator('es-mx-sample.json')

    samples()
    data = compute_TextModel_vocabulary('es-mx-sample.json',
                                        pretrained=False,
                                        token_max_filter=20)
    assert len(data['vocabulary']['dict']) == 20
    data = compute_TextModel_vocabulary(iterator,
                                        pretrained=False,
                                        token_max_filter=20)
    assert len(data['vocabulary']['dict']) == 20
    assert data['vocabulary']['update_calls'] == 2048


def test_compute_SeqTM_vocabulary():
    """test SeqTM vocabulary"""
    samples()
    params = compute_TextModel_vocabulary('es-mx-sample.json',
                                          pretrained=False)
    data = compute_SeqTM_vocabulary('es-mx-sample.json',
                                    params,
                                    pretrained=False,
                                    token_max_filter=2**13)
    assert len(data['vocabulary']['dict']) == 2**13
