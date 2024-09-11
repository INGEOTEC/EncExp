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
from urllib import request
from urllib.error import HTTPError
try:
    USE_TQDM = True
    from tqdm import tqdm
except ImportError:
    USE_TQDM = False


class Download(object):
    """Download
    
    >>> from EncExp.utils import Download
    >>> d = Download("http://github.com", "t.html")
    """

    def __init__(self, url, output='t.tmp') -> None:
        self._url = url
        self._output = output
        try:
            request.urlretrieve(url, output, reporthook=self.progress)
        except HTTPError as exc:
            self.close()
            raise exc
        self.close()

    @property
    def tqdm(self):
        """tqdm"""

        if not USE_TQDM:
            return None
        try:
            return self._tqdm
        except AttributeError:
            self._tqdm = tqdm(total=self._nblocks, leave=False)
        return self._tqdm

    def close(self):
        """Close tqdm if used"""
        if USE_TQDM:
            self.tqdm.close()

    def update(self):
        """Update tqdm if used"""
        if USE_TQDM:
            self.tqdm.update()

    def progress(self, nblocks, block_size, total):
        """tqdm progress"""

        self._nblocks = total // block_size
        self.update()