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
import re
from tokenizers import Tokenizer, pre_tokenizers, normalizers
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from microtc.utils import Counter
from encexp.text_repr import Identifier
URL = re.compile(r'https?://\S+|\b_url\b')
USER = re.compile(r'@\S+|\b_usr\b')


class SeqHF(Identifier):
    """
    SeqHF
    """
    def __init__(self,
                 lang:str='es',
                 vocab_size: int=int(2**15),
                 lc: bool=True,
                 del_diac: bool=True,
                 pretrained: bool=True):
        super().__init__()
        self.lang = lang
        self._text = 'text'
        self.vocab_size = vocab_size
        self.lc = lc
        self.del_diac = del_diac
        self.pretrained = pretrained
        self._is_fitted = False

    def get_text(self, text):
        """Return self._text key from text

        :param text: Text
        :type text: dict
        """
        if isinstance(text, dict):
            text = text[self._text]
        text = re.sub(URL, '[URL]', text)
        text = re.sub(USER, '[USR]', text)
        return text

    @property
    def tokenizer(self):
        """
        Tokenizer
        """
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    def normalizer_params(self):
        """
        Parameters used by the normalizer
        """

        norm_seq_params = [NFD()]
        if self.lc:
            norm_seq_params.append(Lowercase())
        if self.del_diac:
            norm_seq_params.append(StripAccents())
        # rep = normalizers.Replace(Regex(r'https?://\S+'), '[URL]')
        # norm_seq_params.append(rep)
        # norm_seq_params.append(normalizers.Replace(Regex(r'\b_url\b'), '[URL]'))
        return normalizers.Sequence(norm_seq_params)

    def pre_tokenizers_params(self):
        """
        Parameters used by the normalizer
        """
        return pre_tokenizers.Whitespace()
        # _ = pre_tokenizers.Split('[URL]', behavior='isolated')
        # return pre_tokenizers.Sequence([_, pre_tokenizers.Whitespace()])

    def fit_tokenizer(self, X):
        """
        Fit tokenizer
        """
        if self._is_fitted:
            return self
        special_tokens = ['[UNK]', '[URL]', '[USR]']
        self.tokenizer = Tokenizer(WordPiece(unk_token=special_tokens[0]))
        self.tokenizer.add_special_tokens(special_tokens)
        trainer = WordPieceTrainer(vocab_size=self.vocab_size,
                                   special_tokens=special_tokens)
        self.tokenizer.normalizer = self.normalizer_params()
        self.tokenizer.pre_tokenizer = self.pre_tokenizers_params()
        self.tokenizer.train_from_iterator(map(self.get_text, X),
                                           trainer=trainer,
                                           length=len(X))
        return self
    
    def fit_doc_freq(self, X):
        """
        Compute the document frequency
        """
        cnt = Counter()
        for text in X:
            cnt.update(set(self.tokenizer.encode(self.get_text(text)).ids))
        self.doc_freq = cnt
        self._is_fitted = True

    def fit(self, X):
        """
        fit
        
        :param X: Dataset
        """
        self.fit_tokenizer(X)
        self.fit_doc_freq(X)
        return self

    @property
    def doc_freq(self):
        return self._doc_freq

    @doc_freq.setter
    def doc_freq(self, value):
        self._doc_freq = value

    def from_dict(self, data:dict):
        """
        Restore SeqHF from_dict
        
        :param data: dictionary containing the model
        """
        self.doc_freq = Counter(data["dict"],
                                data["update_calls"])
        self.tokenizer = Tokenizer.from_str(data['tokenizer'])
        self._is_fitted = True
        return self

    def to_dict(self):
        """
        Store model in a json
        
        :param self: Descripción
        """
        return dict(update_calls=self.doc_freq.update_calls,
                    dict=dict(self.doc_freq),
                    tokenizer=self.tokenizer.to_str())
    
    def __sklearn_is_fitted__(self):
        """
        Test whether the model is fitted
        """
        return self._is_fitted
