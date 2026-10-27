"""BM25 over a local corpus of text files (numpy only).

Terms are lower-cased \\w+ tokens (no stemming, a short stop list), hashed (crc32) into 2^25 buckets; postings are
stored term-major (CSR): ptr[bucket] .. ptr[bucket + 1] index document ids (int32) and term frequencies (uint16).
Scores are Okapi BM25 with k1 = 0.9 and b = 0.4, the Anserini / BrowseComp-Plus defaults.

  delegent index --corpus corpus/ --out index/
"""
import json
import multiprocessing
from pathlib import Path
import re
import time
import zlib

import numpy as np

BUCKETS = 1 << 25
TOKEN = re.compile(r'\w+', re.UNICODE)
STOP = set('a an and are as at be but by for from has have he her his i if in into is it its of on or she such that the '
           'their them then there these they this to was were which while who will with you your we our not no'.split())
K1, B = 0.9, 0.4


def tokens(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP and len(t) <= 40]


def bucket(term, buckets=BUCKETS):
    return zlib.crc32(term.encode('utf-8')) & (buckets - 1)


def _shard(args):
    paths, buckets = args
    docs, terms, tfs, lengths = [], [], [], []
    for doc_id, path in paths:
        try:
            text = path.read_text(errors='replace')
        except OSError:
            text = ''
        toks = tokens(text)
        lengths.append((doc_id, len(toks)))
        if not toks:
            continue
        ids = np.fromiter((bucket(t, buckets) for t in toks), dtype=np.int64, count=len(toks))
        uniq, counts = np.unique(ids, return_counts=True)
        docs.append(np.full(len(uniq), doc_id, dtype=np.int32))
        terms.append(uniq.astype(np.int32))
        tfs.append(np.minimum(counts, 65535).astype(np.uint16))
    cat = (lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt))
    return cat(docs, np.int32), cat(terms, np.int32), cat(tfs, np.uint16), lengths


def build(corpus, out, workers=None, buckets=BUCKETS):
    corpus, out = Path(corpus), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in corpus.rglob('*') if p.is_file())
    rel = [str(p.relative_to(corpus)) for p in paths]
    print(f'{len(paths)} documents', flush=True)
    started = time.time()
    chunks = [(list(enumerate(paths))[i:i + 500], buckets) for i in range(0, len(paths), 500)]
    doc_len = np.zeros(len(paths), dtype=np.int32)
    parts_d, parts_t, parts_f = [], [], []
    with multiprocessing.Pool(workers) as pool:
        for k, (d, t, f, lengths) in enumerate(pool.imap_unordered(_shard, chunks), 1):
            parts_d.append(d); parts_t.append(t); parts_f.append(f)
            for doc_id, n in lengths:
                doc_len[doc_id] = n
            if k % 20 == 0:
                print(f'{k}/{len(chunks)} shards, {time.time() - started:.0f}s', flush=True)
    docs, terms, tfs = np.concatenate(parts_d), np.concatenate(parts_t), np.concatenate(parts_f)
    del parts_d, parts_t, parts_f
    print(f'{len(docs)} postings; sorting', flush=True)
    order = np.argsort(terms, kind='stable')
    docs, tfs, terms = docs[order], tfs[order], terms[order]
    del order
    counts = np.bincount(terms, minlength=buckets).astype(np.int64)
    ptr = np.zeros(buckets + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    np.save(out / 'ptr.npy', ptr)
    np.save(out / 'docs.npy', docs)
    np.save(out / 'tfs.npy', tfs)
    np.save(out / 'doc_len.npy', doc_len)
    (out / 'paths.json').write_text(json.dumps(rel))
    (out / 'meta.json').write_text(json.dumps({'documents': len(rel), 'postings': int(len(docs)), 'buckets': buckets,
                                               'k1': K1, 'b': B, 'seconds': round(time.time() - started)}))
    print(f'done in {time.time() - started:.0f}s', flush=True)


class Bm25Index:
    def __init__(self, path):
        path = Path(path)
        self.ptr = np.load(path / 'ptr.npy', mmap_mode='r')
        self.docs = np.load(path / 'docs.npy', mmap_mode='r')
        self.tfs = np.load(path / 'tfs.npy', mmap_mode='r')
        self.doc_len = np.load(path / 'doc_len.npy').astype(np.float32)
        self.paths = json.loads((path / 'paths.json').read_text())
        self.buckets = json.loads((path / 'meta.json').read_text()).get('buckets', BUCKETS)
        self.n = len(self.paths)
        self.avg = float(self.doc_len.mean()) or 1.0
        self.norm = K1 * (1 - B + B * self.doc_len / self.avg)

    def search(self, query, k=10):
        scores = np.zeros(self.n, dtype=np.float32)
        hit = False
        for term in dict.fromkeys(tokens(query)):
            b = bucket(term, self.buckets)
            lo, hi = int(self.ptr[b]), int(self.ptr[b + 1])
            if hi <= lo:
                continue
            hit = True
            d = np.asarray(self.docs[lo:hi])
            tf = np.asarray(self.tfs[lo:hi]).astype(np.float32)
            idf = np.log(1 + (self.n - (hi - lo) + 0.5) / ((hi - lo) + 0.5))
            scores += np.bincount(d, weights=idf * tf * (K1 + 1) / (tf + self.norm[d]), minlength=self.n).astype(np.float32)
        if not hit:
            return []
        k = min(k, self.n)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(float(scores[i]), self.paths[i]) for i in top if scores[i] > 0]


def term_regex(query):
    """One compiled alternation of the query's terms (whole words), or None."""
    terms = sorted(set(tokens(query)), key=len, reverse=True)
    if not terms:
        return None
    return re.compile(r'\b(?:' + '|'.join(re.escape(t) for t in terms) + r')\b', re.IGNORECASE)


def best_window(text, query, width=400, limit=400000):
    """The passage of `text` (about `width` characters) with the most distinct query terms (one regex pass)."""
    pattern = term_regex(query)
    if pattern is None:
        return text[:width]
    hits = [(m.start(), m.group(0).lower()) for m in pattern.finditer(text, 0, min(len(text), limit))]
    if not hits:
        return text[:width]
    best, best_n, j = hits[0][0], 0, 0
    for i in range(len(hits)):   # windows starting at a hit; hits are sorted
        while j < len(hits) and hits[j][0] < hits[i][0] + width:
            j += 1
        n = len({t for _, t in hits[i:j]})
        if n > best_n:
            best, best_n = hits[i][0], n
    start = max(0, best - width // 4)
    return text[start:start + width]


def chunk_hits(text, query, size):
    """Per chunk of `size` characters: the number of distinct query terms in it."""
    pattern = term_regex(query)
    counts = [set() for _ in range((len(text) + size - 1) // size or 1)]
    if pattern is not None:
        for m in pattern.finditer(text):
            counts[m.start() // size].add(m.group(0).lower())
    return [len(c) for c in counts]
