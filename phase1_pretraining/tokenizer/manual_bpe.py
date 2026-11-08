import json, heapq, time
from collections import Counter, defaultdict
try:
    import regex as re          # Colab has it; needed for \p{L} and \p{N}
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
except ImportError:
    import re
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
PAT = re.compile(PAT)

def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
B2U = bytes_to_unicode()
SPECIALS = ["<|endoftext|>", "[PAD]"]

def pretokenize(text):
    return ["".join(B2U[b] for b in w.encode("utf-8")) for w in PAT.findall(text)]

def train_bpe(texts, vocab_size=32000, min_frequency=2):
    wc = Counter()
    for t in texts:
        wc.update(pretokenize(t))
    alphabet = sorted({c for w in wc for c in w})
    vocab = {s: i for i, s in enumerate(SPECIALS)}
    for c in alphabet: vocab[c] = len(vocab)
    words = [[vocab[c] for c in w] for w in wc]
    freqs = list(wc.values())
    pair_cnt = defaultdict(int); where = defaultdict(set)
    for wi, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_cnt[p] += freqs[wi]; where[p].add(wi)
    heap = [(-c, p) for p, c in pair_cnt.items()]; heapq.heapify(heap)
    id2tok = {i: s for s, i in vocab.items()}; merges = []
    while len(vocab) < vocab_size and heap:
        negc, pair = heapq.heappop(heap)
        if pair_cnt.get(pair, 0) != -negc:
            if pair_cnt.get(pair, 0) > 0: heapq.heappush(heap, (-pair_cnt[pair], pair))
            continue
        if -negc < min_frequency: break
        a, b = pair; new_tok = id2tok[a] + id2tok[b]
        new_id = len(vocab); vocab[new_tok] = new_id; id2tok[new_id] = new_tok
        merges.append((id2tok[a], id2tok[b]))
        touched = set()
        for wi in list(where[pair]):
            w, f = words[wi], freqs[wi]
            for p in zip(w, w[1:]):
                pair_cnt[p] -= f
                if p != pair: touched.add(p)
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b: out.append(new_id); i += 2
                else: out.append(w[i]); i += 1
            words[wi] = out
            for p in zip(out, out[1:]):
                pair_cnt[p] += f; where[p].add(wi); touched.add(p)
        pair_cnt[pair] = 0; del where[pair]
        for p in touched:
            if pair_cnt[p] > 0: heapq.heappush(heap, (-pair_cnt[p], p))
    return vocab, merges

class ByteBPE:
    def __init__(self, vocab, merges):
        self.vocab, self.id2tok = vocab, {i: t for t, i in vocab.items()}
        self.rank = {tuple(m): r for r, m in enumerate(merges)}; self.cache = {}
    def _bpe(self, w):
        if w in self.cache: return self.cache[w]
        parts = list(w)
        while len(parts) > 1:
            best = min(((self.rank.get(p, 1 << 30), i) for i, p in enumerate(zip(parts, parts[1:]))))
            if best[0] == 1 << 30: break
            i = best[1]; parts[i:i + 2] = [parts[i] + parts[i + 1]]
        self.cache[w] = [self.vocab.get(p, 0) for p in parts]
        return self.cache[w]   # unseen byte symbol maps to unk id 0, like HF
    def get_vocab_size(self): return len(self.vocab)
    def token_to_id(self, t): return self.vocab.get(t)
    def encode(self, text):
        class Enc: pass
        e = Enc(); e.ids = self.encode_ids(text); return e
    def encode_ids(self, text):
        ids = []
        pieces = [text]
        for s in SPECIALS:
            pieces = [x for p in pieces for x in (sum(([q, s] for q in p.split(s)), [])[:-1] if s in p else [p])]
        for p in pieces:
            if p in SPECIALS: ids.append(self.vocab[p])
            else:
                for w in pretokenize(p): ids += self._bpe(w)
        return ids
    def decode(self, ids):
        u2b = {v: k for k, v in B2U.items()}; out = bytearray()
        for i in ids:
            t = self.id2tok[i]
            if t in SPECIALS: out += t.encode()
            else: out += bytes(u2b[c] for c in t)
        return out.decode("utf-8", errors="replace")
    @classmethod
    def from_hf_json(cls, path):
        m = json.load(open(path))["model"]
        return cls(m["vocab"], [tuple(x.split(" ")) if isinstance(x, str) else tuple(x) for x in m["merges"]])
    def save(self, path):
        json.dump({"vocab": self.vocab, "merges": [list(m) for m in sorted(self.rank, key=self.rank.get)]}, open(path, "w"))
