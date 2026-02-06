# Copyright 2026 Mario Graff (https://github.com/mgraffg)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from encexp.hugging_tok import SeqHF
from encexp.utils import load_dataset


def test_SeqHF():
    dataset = load_dataset(dataset='dev')
    seq = SeqHF(del_diac=False).fit_tokenizer(dataset[:2**15])
    seq.fit_doc_freq(dataset)
    assert seq.doc_freq.most_common()[0] == (2413, 50522)


def test_SeqHF_to_dict():
    dataset = load_dataset(dataset='dev')
    seq = SeqHF(del_diac=False).fit_tokenizer(dataset[:2**15])
    seq.fit_doc_freq(dataset)
    model = seq.to_dict()
    assert 'tokenizer' in model
    assert 'dict' in model
    assert 'update_calls' in model
    seq2 = SeqHF(del_diac=False).from_dict(model)
    assert len(seq2.doc_freq) == len(seq.doc_freq)
    
