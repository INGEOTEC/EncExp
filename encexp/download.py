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
import os
from os.path import isdir, isfile, join
from microtc.utils import tweet_iterator
from encexp.utils import Download, MODELS, EncExp_URL


def download(identifier: str, first: bool=True,
             base_url: str=EncExp_URL,
             outputdir: str=MODELS,
             return_path: bool=False):
    """download"""
    if not isdir(outputdir):
        os.mkdir(outputdir)
    output = join(outputdir, f'{identifier}.json.gz')
    if isfile(output):
        if return_path:
            return output
        try:
            if first:
                return next(tweet_iterator(output))
            return tweet_iterator(output)
        except Exception:
            os.unlink(output)
    Download(base_url + f'/{identifier}.json.gz', output)
    if return_path:
        return output
    if first:
        return next(tweet_iterator(output))
    return tweet_iterator(output)


def download_TextModel(identifier: str, first: bool=True):
    """Download TextModel"""
    return download(identifier, first=first)
