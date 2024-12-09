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
from encexp.utils import Download, MODELS, EncExp_URL
from microtc.utils import tweet_iterator
from os.path import isdir, isfile, join
import numpy as np
import os
import encexp


def download_seqtm(lang, voc_size_exponent: int=13,
                   output=None, voc_source='noGeo',
                   prefix='seqtm',
                   prefix_suffix: bool=True):
    """Download SeqTM vocabulary"""
    if not isdir(MODELS):
        os.mkdir(MODELS)
    flags = []
    if prefix_suffix:
        flags.append('ix')
    for flag in [voc_source]:
        if flag is not None:
            flags.append(flag)
    voc_fname = f'{prefix}_{"_".join(flags)}_{lang}_{voc_size_exponent}.json.gz'
    if output is None:
        output = join(MODELS, voc_fname)
    if isfile(output):
        try:
            return next(tweet_iterator(output))
        except Exception as exp:
            os.unlink(output)
    Download(EncExp_URL + f'/{voc_fname}', output)
    return next(tweet_iterator(output))


def download_encexp(lang='es', voc_size_exponent: int=13,
                    precision=np.float16, voc_source='noGeo',
                    enc_source='mix', output=None,
                    prefix_suffix=True,
                    intercept=False):
    """Download EncExp"""
    def read(output):
        iter_ = tweet_iterator(output)
        params = next(iter_)
        coefs = []
        for coef in iter_:
            _ = np.frombuffer(bytearray.fromhex(coef['coef']),
                              dtype=precision)
            coef['coef'] = _
            coefs.append(coef)
        return dict(seqtm=params, coefs=coefs)

    if not isdir(MODELS):
        os.mkdir(MODELS)
    flags = []
    if prefix_suffix:
        flags.append('ix')
    for flag in [voc_source, enc_source]:
        if flag is not None:
            flags.append(flag)
    if intercept:
        flags.append('W0')
    voc_fname = f'encexp_{"_".join(flags)}_{lang}_{voc_size_exponent}.json.gz'
    if output is None:
        output = join(MODELS, voc_fname)
    if isfile(output):
        try:
            return read(output)
        except Exception:
            os.unlink(output)
    assert precision.__name__ == 'float16'
    Download(EncExp_URL + f'/{voc_fname}', output)
    return read(output)


def main(args):
    voc_size_exponent = args.voc_size_exponent
    lang = args.lang
    output = args.output
    if args.seqtm:
        download_seqtm(lang=lang, voc_size_exponent=voc_size_exponent,
                       output=output)
    if args.encexp:
        voc_source = args.voc_source
        enc_source = args.enc_source
        precision = np.float16
        prefix_suffix = args.prefix_suffix
        if voc_source is not None:
            precision = np.float16

        download_encexp(lang=lang,
                        voc_size_exponent=voc_size_exponent,
                        precision=precision,
                        voc_source=voc_source,
                        enc_source=enc_source,
                        output=output,
                        prefix_suffix=prefix_suffix)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download',
                                     prog='EncExp.download')
    parser.add_argument('-v', '--version', action='version',
                        version=f'EncExp {encexp.__version__}')
    parser.add_argument('-o', '--output',
                        help='Output filename',
                        dest='output', default=None, type=str)
    parser.add_argument('--lang', help='Language (ar | ca | de | en | es | fr | hi | in | it | ko | nl | pl | pt | ru | tl | tr )',
                        type=str, default='es')
    parser.add_argument('--voc_size_exponent',
                        help='Vocabulary size express as log2',
                        dest='voc_size_exponent',
                        type=int, default=13)
    parser.add_argument('--voc_source',
                        help='Vocabulary Source', dest='voc_source',
                        default='mix')
    parser.add_argument('--voc_enc',
                        help='Embedding Source', dest='enc_source',
                        default=None)    
    parser.add_argument('--SeqTM',
                        help='Download SeqTM vocabulary',
                        dest='seqtm', action='store_true')
    parser.add_argument('--EncExp',
                        help='Download EncExp',
                        dest='encexp', action='store_true')
    parser.add_argument('--prefix-suffix',
                        help='Restric to use prefix and suffix',
                        dest='prefix_suffix', action='store_true')    
    main(parser.parse_args())