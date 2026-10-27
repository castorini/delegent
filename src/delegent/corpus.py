"""Corpora that the search agent and its subagents read: a local directory of text files with a BM25 index, or the
ClimbMix BM25 search API. Both offer the same two operations, search(query) and doc(docid)."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import threading
import time

import requests


class LocalCorpus:
    """Text files under `root`, searched with a BM25 index built by `delegent index`. A docid is a path relative to
    `root`, shown to the models as /corpus/<path>."""
    prefix = '/corpus/'
    docid_pattern = re.compile(r'/corpus/[^\s)"\'\]>,]+')

    def __init__(self, root, index):
        from .bm25 import Bm25Index
        self.root = Path(root)
        self.index = Bm25Index(index)

    def search(self, query, hits=10):
        out = []
        for score, rel in self.index.search(query, hits):
            try:
                out.append((self.prefix + rel, score, (self.root / rel).read_text(errors='replace')))
            except OSError:
                continue
        return out

    def resolve(self, docid):
        docid = str(docid).strip().strip('"\'<>')
        rel = docid.split(self.prefix, 1)[1] if self.prefix in docid else docid.lstrip('/')
        if (self.root / rel).is_file():
            return rel
        name = Path(rel).name.lower()   # a file name the model shortened or re-typed
        return next((p for p in self.index.paths if Path(p).name.lower() == name), None)

    def doc(self, docid):
        rel = self.resolve(docid)
        return (self.root / rel).read_text(errors='replace') if rel else ''

    def canonical(self, docid):
        rel = self.resolve(docid)
        return self.prefix + rel if rel else None


class ClimbMix:
    """The ClimbMix BM25 search API. A shared service: requests in flight are capped per process, 429/5xx answers are
    retried with jittered backoff, and searches and documents are cached on disk."""
    docid_pattern = re.compile(r'shard_\d+_\d+')
    SEARCH_DOC_CHARS = 30000

    def __init__(self, api=None, token=None, cache=None, max_inflight=3):
        self.api = (api or os.environ.get('CLIMBMIX_API', 'https://api.castorini.uwaterloo.ca/v1/climbmix-400b')).rstrip('/')
        token = token or os.environ.get('CLIMBMIX_TOKEN')
        if not token and os.environ.get('CLIMBMIX_TOKEN_FILE'):
            token = Path(os.environ['CLIMBMIX_TOKEN_FILE']).read_text().strip()
        self.session = requests.Session()
        if token:
            self.session.headers['Authorization'] = 'Bearer ' + token
        self.cache = Path(cache or os.environ.get('CLIMBMIX_CACHE', Path.home() / '.cache/delegent/climbmix'))
        self.inflight = threading.Semaphore(max_inflight)
        self.random = random.Random()
        self.docs, self.lock = {}, threading.Lock()

    def _path(self, kind, key):
        h = hashlib.sha1(key.encode()).hexdigest()
        return self.cache / kind / h[:2] / f'{h}.json'

    def _cached(self, kind, key):
        try:
            return json.loads(self._path(kind, key).read_text())
        except (OSError, ValueError):
            return None

    def _store(self, kind, key, value):
        path = self._path(kind, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f'.tmp{os.getpid()}{threading.get_ident()}')
            tmp.write_text(json.dumps(value))
            os.replace(tmp, path)
        except OSError:
            pass

    def get(self, path, params, patience=3600):
        start, delay = time.time(), 2.0
        while True:
            with self.inflight:
                try:
                    r = self.session.get(self.api + path, params=params, timeout=(20, 120))
                    if r.status_code == 200:
                        return r.json()
                    if r.status_code < 500 and r.status_code != 429:
                        raise RuntimeError(f'ClimbMix API {r.status_code}: {r.text[:200]}')
                    retry_after = float(r.headers.get('Retry-After') or 0)
                except (requests.ConnectionError, requests.Timeout):
                    retry_after = 0
            if time.time() - start > patience:
                raise RuntimeError(f'ClimbMix API unavailable for {patience}s: {path}')
            time.sleep(max(retry_after, delay) * (0.5 + self.random.random()))
            delay = min(delay * 2, 60)

    def search(self, query, hits=10):
        query = ' '.join(str(query).split())[:600]   # very long URLs are rejected
        key = f'{hits}\t{query}'
        data = self._cached('search', key)
        if data is None:
            data = self.get('/search', {'query': query, 'hits': hits, 'max_doc_length': self.SEARCH_DOC_CHARS})
            data = {'candidates': [{'docid': c['docid'], 'score': c.get('score'), 'doc': c.get('doc') or ''}
                                   for c in data.get('candidates', [])]}
            self._store('search', key, data)
        return [(c['docid'], c.get('score'), c.get('doc') or '') for c in data['candidates']]

    def doc(self, docid):
        m = self.docid_pattern.search(str(docid))
        if not m:
            return ''
        docid = m.group(0)
        with self.lock:
            if docid in self.docs:
                return self.docs[docid]
        data = self._cached('docs', docid)
        if data is None:
            try:
                data = {'doc': self.get(f'/doc/{docid}', {'max_doc_length': 2000000}).get('doc') or ''}
            except RuntimeError:   # unknown docid
                return ''
            self._store('docs', docid, data)
        with self.lock:
            if len(self.docs) > 5000:
                self.docs.clear()
            self.docs[docid] = data['doc']
        return data['doc']

    def canonical(self, docid):
        m = self.docid_pattern.search(str(docid))
        return m.group(0) if m else None


def open_corpus(kind, root=None, index=None):
    if kind == 'climbmix':
        return ClimbMix()
    if kind == 'local':
        if not (root and index):
            raise ValueError('a local corpus needs --corpus-root and --index')
        return LocalCorpus(root, index)
    raise ValueError(f'unknown corpus {kind!r}')
