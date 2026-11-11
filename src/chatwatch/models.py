"""Optional local ML scorers (T1 tier): CPU, ONNX int8, no PyTorch needed.

Toxicity: Detoxify's unbiased-toxic-roberta (Jigsaw Unintended Bias), int8
ONNX build. research/03 measured ~1.3 ms/message on this CPU. Its scores
are *features* for the archetype layer, never verdicts: on Twitch chat it
flags praise ("you killed it") and misses "kys" and bait.

Weights are downloaded once into models/ (inside the project folder).
"""
import os

import numpy as np

MODEL_DIR = os.environ.get("CW_MODELS", "models")
# Both encoders are downloaded on first use from Hugging Face, pinned to the commit chatwatch was measured with, and
# every file is checked against its sha256 (a file that does not match is not used).
TOX_REPO = "protectai/unbiased-toxic-roberta-onnx"
TOX_REV = "16fd63c0e51000f407fb7d28ce655e41495ebc8b"
TOX_FILES = ["config.json", "model_quantized.onnx", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]


SHA256 = {
    "unbiased-toxic-roberta-onnx/config.json": "2811061b384e325839ce61586c97a30f6b287acbd18caec1db125f5b14484cac",
    "unbiased-toxic-roberta-onnx/model_quantized.onnx": "2747e65b2db4071959a2c4c645dd54bf9ff910b4fef66929f5fd450c4976a280",
    "unbiased-toxic-roberta-onnx/tokenizer.json": "2f1a555a1ee93656b4e6f67aa75d492a843c225e5ef754bae24c36bd85851cd7",
    "unbiased-toxic-roberta-onnx/tokenizer_config.json": "e66f9d5dfa61fc8f8442242601e21569f93bec58b47c11a8eb5f49d47fba48cd",
    "unbiased-toxic-roberta-onnx/special_tokens_map.json": "8293ae960b0a0852d4d3813118030a1149a3ed9fe37bc2e1e7b3c2e62eb2d4b7",
    "unbiased-toxic-roberta-onnx/vocab.json": "ed19656ea1707df69134c4af35c8ceda2cc9860bf2c3495026153a133670ab5e",
    "unbiased-toxic-roberta-onnx/merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "all-MiniLM-L6-v2/config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "all-MiniLM-L6-v2/tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "all-MiniLM-L6-v2/onnx/model_qint8_avx512_vnni.onnx": "4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474",
}


_CHECKED = set()


def _sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _fetch(repo, files, dest, rev="main"):
    """Download the model's files (once) at the pinned revision; each is checked against SHA256 before use."""
    import urllib.request
    os.makedirs(dest, exist_ok=True)
    for f in files:
        p = os.path.join(dest, f)
        key = f"{os.path.basename(dest)}/{f}"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if os.path.exists(p):                 # on disk already: checked once per run
            if key in SHA256 and key not in _CHECKED and _sha256(p) != SHA256[key]:
                raise RuntimeError(f"{p} does not match its pinned sha256: delete it and start again to download it")
            _CHECKED.add(key)
        else:
            url = f"https://huggingface.co/{repo}/resolve/{rev}/{f}"
            print(f"[models] downloading {repo} {f} ...", flush=True)
            urllib.request.urlretrieve(url, p + ".part")
            if key in SHA256 and _sha256(p + ".part") != SHA256[key]:
                os.remove(p + ".part")
                raise RuntimeError(f"{repo} {f}: the download does not match its pinned sha256; not used")
            os.replace(p + ".part", p)
    return dest


class Toxicity:
    # label -> feature name used by archetypes.message_evidence
    FEATURES = {"toxicity": "ml_toxicity", "severe_toxicity": "ml_severe",
                "obscene": "ml_obscene", "threat": "ml_threat", "insult": "ml_insult",
                "identity_attack": "ml_identity_attack", "sexual_explicit": "ml_sexual"}

    def __init__(self, threads=4):
        import json

        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = _fetch(TOX_REPO, TOX_FILES, os.path.join(MODEL_DIR, "unbiased-toxic-roberta-onnx"), TOX_REV)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(os.path.join(d, "model_quantized.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        self.tok.no_padding()                 # we pad per batch; see score()
        self.tok.enable_truncation(128)
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.labels = [cfg["id2label"][str(i)] for i in range(len(cfg["id2label"]))]

    def score(self, texts):
        if not texts:
            return []
        enc = self.tok.encode_batch(list(texts))
        n = max(len(e.ids) for e in enc)
        ids = np.ones((len(enc), n), dtype=np.int64)          # roberta pad id = 1
        mask = np.zeros((len(enc), n), dtype=np.int64)
        for i, e in enumerate(enc):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = e.attention_mask
        feed = {"input_ids": ids, "attention_mask": mask}
        feed = {k: v for k, v in feed.items() if k in self.inputs}
        logits = self.sess.run(None, feed)[0]
        probs = 1 / (1 + np.exp(-logits))
        out = []
        for row in probs:
            d = {self.FEATURES[l]: float(p) for l, p in zip(self.labels, row) if l in self.FEATURES}
            # identity *mentions* (not attacks) are their own signal
            d["ml_identity_mention"] = float(max((p for l, p in zip(self.labels, row)
                                                  if l not in self.FEATURES), default=0.0))
            out.append(d)
        return out

    def score_one(self, text):
        return self.score([text])[0]


if __name__ == "__main__":
    import json
    import time
    t = Toxicity()
    probe = ["that play was sick, you killed it", "kys", "who even watches this guy",
             "you're so trash at this game lmao", "hi chat first time here!",
             "is the streamer gay?", "buy followers at cheapviewz dot com",
             "no offense but your voice is really annoying", "stfu foid"]
    t0 = time.time()
    for s, r in zip(probe, t.score(probe)):
        print(f"{s[:45]:<46} " + " ".join(f"{k[3:]}={v:.2f}" for k, v in r.items() if v >= 0.05))
    print(f"batch of {len(probe)} in {(time.time() - t0) * 1000:.0f} ms; labels: {t.labels}")


EMB_REPO = "sentence-transformers/all-MiniLM-L6-v2"
EMB_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EMB_FILES = ["config.json", "tokenizer.json", "onnx/model_qint8_avx512_vnni.onnx"]

# A handful of example messages per archetype. A message close to these (cosine)
# gets modest evidence, which catches paraphrases the regex cues miss
# ("nobody with a brain would sit through this" never says "who watches").
# Benign prototypes exist so similarity is relative, not absolute.
PROTOTYPES = {
    "provocateur": [
        "who even watches this garbage", "this streamer is so bad at the game", "you are so boring",
        "nobody cares about your stream", "this content is trash", "how do you have viewers",
        "unwatchable stream", "you fell off so hard", "your gameplay is embarrassing",
        "imagine paying to watch this", "worst streamer on the platform",
    ],
    "concern_troll": [
        "no offense but your voice is annoying", "not to be rude but why is the stream so boring",
        "genuine question why are you always this bad", "is it true that you got banned for cheating",
        "just asking, why do people even watch you", "with all due respect you should quit",
    ],
    "injector": [
        "what do you think about the election", "thoughts on the israel palestine situation",
        "trump 2028", "feminism ruined gaming", "did you hear what that other streamer said about you",
        "react to the drama", "woke gaming is cancer",
    ],
    "harasser": [
        "you look so hot today", "show us your feet", "marry me please", "you are ugly",
        "kill yourself", "go back to the kitchen", "send pics",
    ],
    "spammer": [
        "best viewers on example dot com", "buy followers cheap", "check out my channel",
        "follow me back", "free skins giveaway click the link", "dm me for promotion",
    ],
    "_benign": [
        "hi chat", "what game is this", "lol that was funny", "nice play", "gg", "love the stream",
        "you should go left", "how long have you been streaming", "that boss was hard",
        "hello from germany", "first time here", "this music is great", "i think the key is upstairs",
        "chat is so funny today", "good morning everyone", "what rank are you",
    ],
}


class Embedder:
    """all-MiniLM-L6-v2, int8 ONNX (AVX-512 VNNI build), mean-pooled + L2."""

    def __init__(self, threads=2):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = _fetch(EMB_REPO, EMB_FILES, os.path.join(MODEL_DIR, "all-MiniLM-L6-v2"), EMB_REV)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(os.path.join(d, EMB_FILES[2]), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        self.tok.no_padding()                 # tokenizer.json ships with pad-to-128
        self.tok.enable_truncation(64)
        self.proto = {a: self.embed(ex) for a, ex in PROTOTYPES.items()}

    def embed(self, texts):
        enc = self.tok.encode_batch(list(texts))
        n = max(len(e.ids) for e in enc)
        ids = np.zeros((len(enc), n), dtype=np.int64)
        mask = np.zeros((len(enc), n), dtype=np.int64)
        for i, e in enumerate(enc):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = e.attention_mask
        feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
        out = self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
        m = mask[..., None].astype(np.float32)
        v = (out * m).sum(1) / np.maximum(m.sum(1), 1e-9)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def proto_scores(self, text, v=None):
        """{archetype: max cosine to its prototypes} plus '_benign'."""
        v = self.embed([text])[0] if v is None else v
        return {a: float((P @ v).max()) for a, P in self.proto.items()}


class Combined:
    """T1 bundle the engine calls: toxicity scores + prototype similarities."""

    def __init__(self, toxicity=True, embed=True, threads=None):
        self.tox = (Toxicity(threads) if threads else Toxicity()) if toxicity else None
        self.emb = (Embedder(threads) if threads else Embedder()) if embed else None

    def score_one(self, text):
        out = self.tox.score_one(text) if self.tox else {}
        if self.emb:
            from .text import normalize
            v = self.emb.embed([normalize(text)])[0]
            out["proto"] = self.emb.proto_scores(text, v)
            out["ml_emb"] = v                 # for the distilled toxicity head
        return out




class ToxCache:
    """Toxicity-model scores shared by every job that re-reads collected chat
    (store update, forward test, refit, study, learning value, step 1, judged-set
    builder, noise check): each text is scored by the toxicity model once, by
    whichever job reads it first. That model is the slow part (memory-bandwidth
    bound, ~600 texts/s however many threads); the embedding is ~10x cheaper
    and is recomputed, so the cache stays small: 44 bytes a text, ~70 MB a day.

    The scores are float32 in the model, so storing them as float32 is exact:
    a cached read gives the same numbers as scoring again (tests).
    data/mlcache/<model fingerprint>/tox_<day>_<writer>.bin, a file per job run
    and day of chat (append-only; readers take whole records only). Days are by
    the message's time; days older than KEEP_DAYS are deleted."""

    KEEP_DAYS = 14
    REC = None

    def __init__(self, tox, root=None):
        import hashlib
        import time
        root = root or os.environ.get("CW_MLCACHE", "data/mlcache")
        model = os.path.join(MODEL_DIR, "unbiased-toxic-roberta-onnx", "model_quantized.onnx")
        h = hashlib.sha256(open(model, "rb").read())
        h.update(",".join(tox.labels).encode())
        self.dir = os.path.join(root, h.hexdigest()[:16])
        os.makedirs(self.dir, exist_ok=True)
        self.keys = [Toxicity.FEATURES[l] for l in tox.labels if l in Toxicity.FEATURES] + ["ml_identity_mention"]
        self.rec = np.dtype([("h", "V12"), ("t", "<f4", (len(self.keys),))])
        self.days, self.out = {}, {}
        self.me = f"{os.getpid()}_{int(time.time())}"
        self.hits = self.misses = 0
        cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - self.KEEP_DAYS * 86400))
        for f in os.listdir(self.dir):
            if f.startswith("tox_") and f[4:14] < cutoff:
                try:
                    os.remove(os.path.join(self.dir, f))
                except OSError:
                    pass

    @staticmethod
    def key(text):
        import hashlib
        return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=12).digest()

    def _day(self, day):
        idx = self.days.get(day)
        if idx is None:
            idx = {}
            for f in sorted(os.listdir(self.dir)):
                if f.startswith(f"tox_{day}_"):
                    raw = open(os.path.join(self.dir, f), "rb").read()
                    a = np.frombuffer(raw[:len(raw) // self.rec.itemsize * self.rec.itemsize], dtype=self.rec)
                    idx.update(zip(a["h"].tolist(), a["t"]))
            if len(self.days) >= 3:                  # a replay moves forward through the days
                self.days.pop(next(iter(self.days)))
            self.days[day] = idx
        return idx

    def get(self, day, h):
        t = self._day(day).get(h)
        if t is None:
            self.misses += 1
            return None
        self.hits += 1
        return dict(zip(self.keys, map(float, t)))

    def put(self, day, h, d):
        t = np.array([d[k] for k in self.keys], dtype="<f4")
        self._day(day)[h] = t
        f = self.out.get(day)
        if f is None:
            f = self.out[day] = open(os.path.join(self.dir, f"tox_{day}_{self.me}.bin"), "ab")
        f.write(np.array([(h, t)], dtype=self.rec).tobytes())

    def flush(self):
        for f in self.out.values():
            f.flush()


class Ahead:
    """Combined for replays (jobs that re-read collected chat): the texts of the
    next few thousand messages are scored by a pool of threads while the engine
    works on the current ones, and recent texts are remembered (chat repeats
    itself). Each text is still scored on its own (a batch of one), so the
    numbers are identical to Combined.score_one (data/scratch/bench_ahead.py).
    Batch-of-one toxicity inference is limited by memory bandwidth (the weights
    are re-read per message): threads sharing one model beat processes with a
    copy each (data/scratch/bench_threads.py), ~4x a plain Combined overall.
    The engine must read through ahead():

        ml = Ahead()
        eng = Engine(ml=ml, emotes=em, ...)
        for ev in ml.ahead(events, em):
            eng.process(ev)

    Without ahead() it is a plain Combined with a memory of recent texts."""

    def __init__(self, workers=None, keep=50_000, cache=True):
        import atexit
        from collections import OrderedDict
        from concurrent.futures import ThreadPoolExecutor
        self.ml = Combined(threads=1)            # shared by the threads: one thread per inference
        self.tox, self.emb = self.ml.tox, self.ml.emb
        self.pool = ThreadPoolExecutor(workers or int(os.environ.get("CW_ML_WORKERS", "12")))
        self.memo, self.keep = OrderedDict(), keep
        self.cache = ToxCache(self.tox) if cache else None
        if self.cache:
            atexit.register(self.cache.flush)

    def _remember(self, text, d):
        self.memo[text] = d
        if len(self.memo) > self.keep:
            self.memo.popitem(last=False)

    def score_one(self, text):
        d = self.memo.get(text)
        if d is None:
            d = self.ml.score_one(text)          # not scored ahead: score it now (same numbers)
            self._remember(text, d)
        else:
            self.memo.move_to_end(text)
        return d

    def _score_many(self, texts):
        return [self.ml.score_one(t) for t in texts]

    def _score_jobs(self, jobs):
        """[(text, toxicity scores from the cache or None)] -> score_one's dicts:
        the embedding part is always computed, the toxicity model only on a miss."""
        out = []
        for text, tox in jobs:
            if tox is None:
                out.append(self.ml.score_one(text))
                continue
            from .text import normalize
            d = dict(tox)
            v = self.emb.embed([normalize(text)])[0]
            d["proto"] = self.emb.proto_scores(text, v)
            d["ml_emb"] = v
            out.append(d)
        return out

    def ahead(self, items, emotes=None, chunk=2000, get=lambda x: x):
        """Yield items unchanged after scoring the texts the engine will ask for
        (messages with 2+ words that aren't commands; get(item) is the event)."""
        import time
        from itertools import islice

        from .signals import COMMAND
        from .text import tokenize

        def submit(batch):
            want = {}
            for it in batch:
                ev = get(it)
                if not ev or ev.get("kind") != "msg":
                    continue
                raw = ev.get("text") or ""
                if raw in self.memo or raw in want or COMMAND.match(raw):
                    continue
                third = emotes.codes(ev.get("room_id")) if emotes else frozenset()
                if len(tokenize(raw, ev.get("emotes", ()), third).words) >= 2:
                    want[raw] = time.strftime("%Y-%m-%d", time.gmtime(ev["ts"] / 1000)) if self.cache else None
            jobs = []
            for raw, day in want.items():
                h = self.cache.key(raw) if self.cache else None
                jobs.append((raw, day, h, self.cache.get(day, h) if self.cache else None))
            return [(jobs[i:i + 64], self.pool.submit(self._score_jobs, [(j[0], j[3]) for j in jobs[i:i + 64]]))
                    for i in range(0, len(jobs), 64)]
        it = iter(items)
        batch = list(islice(it, chunk))
        pending = submit(batch)
        while batch:
            nxt = list(islice(it, chunk))
            nxt_pending = submit(nxt) if nxt else []     # scored while this batch is processed
            for jobs, f in pending:
                for (t, day, h, cached), d in zip(jobs, f.result()):
                    self._remember(t, d)
                    if self.cache and cached is None:
                        self.cache.put(day, h, d)
            yield from batch
            batch, pending = nxt, nxt_pending
