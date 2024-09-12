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

import argparse
import sys
import json
from EncExp.utils import compute_vocabulary
from EncExp.text_repr import SeqTM
import EncExp


def main(args):
    """CLI"""

    def output_json(D, file):
        print(json.dumps(D), file=file)

    filename  = args.file
    lang = args.lang
    data = compute_vocabulary(filename, lang=lang)
    seqtm = SeqTM(vocabulary=data)
    data = compute_vocabulary(filename,
                              tokenize=seqtm.tokenize,
                              params=data['params'])
    if args.output is None:
        output_json(data, file=sys.stdout)
    else:
        with open(args.output, 'w', encoding='utf-8') as fpt:
            output_json(data, file=fpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute SeqTM Vocabulary',
                                     prog='EncExp.build_voc')
    parser.add_argument('-v', '--version', action='version',
                        version=f'EncExp {EncExp.__version__}')
    parser.add_argument('-o', '--output',
                        help='output filename',
                        dest='output', default=None, type=str)
    parser.add_argument('--lang', help='Language (ar | ca | de | en | es | fr | hi | in | it | ja | ko | nl | pl | pt | ru | tl | tr | zh)',
                        type=str, default='es')
    parser.add_argument('file',
                        help='input filename',
                        nargs=1, type=str)
    args = parser.parse_args()
    main(args)