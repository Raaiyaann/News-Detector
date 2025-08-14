# app.py
import os
import re
import html
import time
import unicodedata
from urllib.parse import urlparse
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_cors import CORS

import requests
import joblib

# === Scraper/metadata ===
import trafilatura
from trafilatura.metadata import extract_metadata as tf_extract_metadata

# === (Opsional) Stemming bahasa Indonesia ===
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _stemmer = StemmerFactory().create_stemmer()
except Exception:
    _stemmer = None

# === Sklearn pipeline type-hint (opsional) ===
try:
    from sklearn.pipeline import Pipeline  # type: ignore
except Exception:
    Pipeline = None  # type: ignore

# ---------- Konfigurasi ----------
MODEL_PATH = os.getenv("MODEL_PATH", "best_random_forest_model.pkl")
VECTOR_PATH = os.getenv("VECTOR_PATH", "vectorizer.pkl")  # untuk mode TF-IDF
USER_AGENT = os.getenv("SCRAPER_UA", "Mozilla/5.0 (compatible; HoaxNewsDetector/1.0; +https://example.com/bot)")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "12"))
MIN_TEXT_WORDS = int(os.getenv("MIN_TEXT_WORDS", "50"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "200000"))

# -- Konfigurasi EMBEDDING BERT (cocokkan dgn Colab) --
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "indobenchmark/indobert-base-p1")
EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "512"))
EMBED_CHUNK_TOKENS = int(os.getenv("EMBED_CHUNK_TOKENS", "512"))

# --- Konfigurasi HOAX & Safety ---
HOAX_LABELS = set(x.strip().lower() for x in os.getenv("HOAX_LABELS", "0,hoax,fake").split(","))
HOAX_THRESHOLD = float(os.getenv("HOAX_THRESHOLD", "0.6"))
TRUSTED_DOMAINS = set(
    x.strip().lower()
    for x in os.getenv(
        "TRUSTED_DOMAINS",
        "cnnindonesia.com,kompas.com,detik.com,tempo.co,antaranews.com,bbc.com,cnn.com"
    ).split(",")
)

# ---------- App ----------
app = Flask(__name__)
CORS(app)

# ---------- Stopwords ----------
STOPWORDS_ID = {
    "yang","dan","di","ke","dari","untuk","pada","dengan","atau","karena","sebagai","bahwa",
    "ini","itu","ia","dia","kami","kita","mereka","ada","tidak","bukan","akan","telah","sudah",
    "dalam","juga","agar","harus","bisa","dapat","oleh","para","serta","tanpa","setelah","sebelum",
    "saat","adalah","lebih","kurang","jadi","bahkan","kalau","jika","sehingga","namun","tetapi",
}
STOPWORDS_EN = {
    "the","and","to","of","in","a","for","is","on","that","with","as","at","by","an","be","are","from",
    "it","this","was","were","or","which","but","not","have","has","had","their","its","they","he","she",
    "we","you","your","our","us","them","his","her",
}
STOPWORDS = STOPWORDS_ID | STOPWORDS_EN

# ---------- Utils: scraping & metadata ----------
def fetch_html(url: str) -> str:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    ctype = r.headers.get("Content-Type", "")
    if "text/html" not in ctype:
        raise ValueError(f"Konten bukan HTML (Content-Type: {ctype})")
    return r.text

def _safe_get(meta, key):
    if not meta:
        return None
    if hasattr(meta, key):
        return getattr(meta, key)
    if isinstance(meta, dict):
        return meta.get(key)
    return None

def _fallback_meta_from_html(html_str: str):
    m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]*content=["\'](.*?)["\']', html_str, re.I)
    title = html.unescape(m.group(1).strip()) if m else None
    if not title:
        m2 = re.search(r'<title[^>]*>(.*?)</title>', html_str, re.I | re.S)
        if m2:
            title = html.unescape(re.sub(r'\s+', ' ', m2.group(1)).strip())
    d = re.search(r'<meta[^>]+property=["\']article:published_time["\'][^>]*content=["\'](.*?)["\']', html_str, re.I)
    published = d.group(1).strip() if d else None
    return title, published

def extract_article(url: str):
    html_str = fetch_html(url)
    text = trafilatura.extract(
        html_str,
        include_comments=False,
        include_tables=False,
        deduplicate=True,
        favor_recall=True,
    )
    try:
        meta = tf_extract_metadata(html_str, url=url)
    except TypeError:
        meta = tf_extract_metadata(html_str)
    except Exception:
        meta = None
    title = _safe_get(meta, "title")
    published = _safe_get(meta, "date")
    if not title or not published:
        f_title, f_published = _fallback_meta_from_html(html_str)
        title = title or f_title
        published = published or f_published
    return (text or "")[:MAX_CHARS], title or "", published or ""

# ---------- Utils: preprocessing ----------
def normalize_text(s: str) -> str:
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00ad", "")
    s = s.lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"[^a-zA-Z\u00C0-\u024F\u1E00-\u1EFF\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = [t for t in s.split() if t not in STOPWORDS and len(t) > 2]
    s = " ".join(tokens)
    if _stemmer:
        try:
            s = _stemmer.stem(s)
        except Exception:
            pass
    return s

# --- Penentuan hoaks yang bisa dikonfigurasi ---
def decide_hoax(pred_label, classes=None, proba=None):
    """
    Mengembalikan (is_hoax: bool, hoax_prob: Optional[float]).
    - is_hoax True jika label prediksi termasuk HOAX_LABELS & (prob hoaks >= HOAX_THRESHOLD jika tersedia),
      atau jika ada kelas hoaks di classes dengan probabilitas >= HOAX_THRESHOLD.
    """
    pl = str(pred_label).strip().lower()
    hoax_prob = None
    if classes is not None and proba is not None:
        for k, p in zip(classes, proba):
            if str(k).strip().lower() in HOAX_LABELS:
                hoax_prob = float(p)
                break

    if pl in HOAX_LABELS:
        if hoax_prob is not None:
            return (hoax_prob >= HOAX_THRESHOLD), hoax_prob
        return True, None

    if hoax_prob is not None:
        return (hoax_prob >= HOAX_THRESHOLD), hoax_prob

    return False, None

# ---------- Model & embedding loader ----------
class Predictor:
    def __init__(self, model_obj, vectorizer=None, is_pipeline=False, embedder=None):
        self.model_obj = model_obj
        self.vectorizer = vectorizer
        self.is_pipeline = is_pipeline
        self.embedder = embedder  # callable: str -> np.ndarray[1, D]

    def predict(self, clean_text: str):
        if self.is_pipeline:
            y = self.model_obj.predict([clean_text])[0]
            proba = self.model_obj.predict_proba([clean_text])[0] if hasattr(self.model_obj, "predict_proba") else None
            classes = getattr(self.model_obj, "classes_", None)
            return y, proba, classes

        if self.vectorizer is not None:
            X = self.vectorizer.transform([clean_text])
            y = self.model_obj.predict(X)[0]
            proba = self.model_obj.predict_proba(X)[0] if hasattr(self.model_obj, "predict_proba") else None
            classes = getattr(self.model_obj, "classes_", None)
            return y, proba, classes

        if self.embedder is not None:
            X = self.embedder(clean_text)  # shape: (1, D)
            y = self.model_obj.predict(X)[0]
            proba = self.model_obj.predict_proba(X)[0] if hasattr(self.model_obj, "predict_proba") else None
            classes = getattr(self.model_obj, "classes_", None)
            return y, proba, classes

        raise RuntimeError(
            "Model bukan Pipeline dan tidak ada vectorizer/embedding. "
            "Sediakan VECTOR_PATH (TF-IDF) atau aktifkan EMBED_MODEL_NAME (transformers+torch)."
        )

# -- Embedding IndoBERT (mean last_hidden_state) --
_tokenizer = None
_model = None
_device = None

def _init_embedder():
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None:
        return
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # opsional: hilangkan warning symlink di Windows

    import torch
    from transformers import AutoTokenizer, AutoModel

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, use_fast=True)
    try:
        _model = AutoModel.from_pretrained(EMBED_MODEL_NAME, use_safetensors=True).to(_device)
    except Exception as e:
        raise RuntimeError(
            "Gagal memuat model dengan safetensors. Pastikan paket 'safetensors' terpasang "
            "atau upgrade torch >= 2.6. Detail: " + str(e)
        )
    _model.eval()

def embed_text_mean_last_hidden_state(text: str):
    import numpy as np
    import torch
    _init_embedder()

    words = text.split()
    if not words:
        return np.zeros((1, _model.config.hidden_size), dtype=np.float32)

    chunks = []
    step = 4000  # potongan kata (tokenizer tetap truncation 512)
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i+step]))

    embs = []
    with torch.no_grad():
        for chunk in chunks:
            inputs = _tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=EMBED_MAX_LENGTH,
            )
            inputs = {k: v.to(_model.device) for k, v in inputs.items()}
            outputs = _model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # (1, hidden)
            embs.append(emb)

    import numpy as np
    arr = np.vstack(embs)             # (n, hidden)
    arr = arr.mean(axis=0, keepdims=True).astype("float32")  # (1, hidden)
    return arr

def load_predictor():
    obj = joblib.load(MODEL_PATH)

    # 1) Pipeline?
    if Pipeline is not None and isinstance(obj, Pipeline):
        return Predictor(obj, is_pipeline=True)
    if hasattr(obj, "named_steps"):
        return Predictor(obj, is_pipeline=True)

    # 2) dict/tuple (model + vectorizer)
    if isinstance(obj, dict):
        model_obj = obj.get("model") or obj.get("clf") or obj.get("classifier") or obj
        vectorizer = obj.get("vectorizer") or obj.get("tfidf") or obj.get("vect")
        if hasattr(model_obj, "predict") and (vectorizer is not None):
            return Predictor(model_obj, vectorizer=vectorizer, is_pipeline=False)

    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        a, b = obj
        model_obj = a if hasattr(a, "predict") else b
        vectorizer = b if model_obj is a else a
        if hasattr(model_obj, "predict") and hasattr(vectorizer, "transform"):
            return Predictor(model_obj, vectorizer=vectorizer, is_pipeline=False)

    # 3) Hanya classifier → coba TF-IDF; kalau tidak ada, pakai EMBEDDING BERT
    model_obj = obj
    vectorizer = None
    if os.path.exists(VECTOR_PATH):
        try:
            vectorizer = joblib.load(VECTOR_PATH)
        except Exception:
            vectorizer = None

    if vectorizer is not None:
        return Predictor(model_obj, vectorizer=vectorizer, is_pipeline=False)

    return Predictor(model_obj, vectorizer=None, is_pipeline=False, embedder=embed_text_mean_last_hidden_state)

# Muat predictor saat start
predictor = load_predictor()

# ---------- Routes ----------
@app.route("/health", methods=["GET"])
def health():
    info = {
        "ok": True,
        "model_loaded": True,
        "is_pipeline": bool(getattr(predictor, "is_pipeline", False)),
        "has_vectorizer": predictor.vectorizer is not None,
        "uses_bert_embedder": predictor.embedder is not None,
        "embed_model": EMBED_MODEL_NAME if predictor.embedder else None,
        "hoax_labels": sorted(list(HOAX_LABELS)),
        "hoax_threshold": HOAX_THRESHOLD,
    }
    return jsonify(info)

@app.route("/predict", methods=["POST"])
def predict():
    t0 = time.time()
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"error": "Field 'url' wajib diisi."}), 400

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return jsonify({"error": "URL tidak valid."}), 400
    except Exception:
        return jsonify({"error": "URL tidak valid."}), 400

    try:
        raw_text, title, published = extract_article(url)
        if not raw_text:
            return jsonify({"error": "Tidak berhasil mengekstrak teks artikel dari URL ini."}), 422

        clean_text = normalize_text(raw_text)
        y_pred, proba, classes = predictor.predict(clean_text)
        pred_label = str(y_pred)

        # pastikan classes tersedia jika ada
        if classes is None and hasattr(predictor.model_obj, "classes_"):
            classes = predictor.model_obj.classes_

        # probabilitas & confidence
        probabilities = None
        confidence = None
        if proba is not None:
            if classes is not None:
                probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
                try:
                    idx = list(map(str, classes)).index(pred_label)
                    confidence = float(proba[idx])
                except Exception:
                    confidence = float(max(proba))
            else:
                probabilities = {str(i): float(p) for i, p in enumerate(proba)}
                confidence = float(max(proba))

        # keputusan hoaks berbasis mapping + threshold
        is_hoax, hoax_prob = decide_hoax(pred_label, classes=classes, proba=proba)

        # safety net untuk domain tepercaya
        domain = urlparse(url).netloc.lower()
        safety_note = None
        if domain in TRUSTED_DOMAINS and is_hoax:
            if hoax_prob is None or hoax_prob < 0.90:
                is_hoax = False
                safety_note = f"Diturunkan karena domain tepercaya ({domain}) dan probabilitas hoaks < 0.90"

        payload = {
            "prediction": pred_label,
            "is_hoax": is_hoax,
            "hoax_probability": hoax_prob,
            "hoax_threshold": HOAX_THRESHOLD,
            "probabilities": probabilities,
            "classes": [str(c) for c in classes] if classes is not None else None,
            "confidence": confidence,  # prob. untuk label yang diprediksi
            "title": title,
            "published": published,
            "source": domain,
            "url": url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "words": len(raw_text.split()),
            "excerpt": raw_text[:600].strip() + ("…" if len(raw_text) > 600 else ""),
            "preprocessed_words": len(clean_text.split()),
            "latency_ms": int((time.time() - t0) * 1000),
            "note": safety_note,
        }
        return jsonify(payload)

    except requests.HTTPError as e:
        return jsonify({"error": f"Gagal mengambil halaman: {e}"}), 502
    except requests.Timeout:
        return jsonify({"error": "Timeout saat mengambil halaman."}), 504
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Gagal memproses: {e}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
