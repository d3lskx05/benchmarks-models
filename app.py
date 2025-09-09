# app.py
import os
import time
import zipfile
import tempfile
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import gdown
import huggingface_hub
import numpy as np
import pandas as pd
import psutil
import streamlit as st

# ML libs
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import onnxruntime as ort

# metrics
from scipy.stats import spearmanr, pearsonr

# ---------------------------------------------------------------------
# ----------------------  Streamlit UI config  ------------------------
# ---------------------------------------------------------------------
st.set_page_config(page_title="Model comparator (Original vs Quant)", layout="wide")
st.title("🔍 Сравнение моделей: Original vs Quantized (легко и понятно)")

# ---------------------------------------------------------------------
# ----------------------  Helpers / small utils  ----------------------
# ---------------------------------------------------------------------
def human_mb(x_bytes: int) -> float:
    return x_bytes / 1024 ** 2

def sample_memory_peak(process: psutil.Process, get_vram_fn=None) -> Tuple[float, float]:
    """Return current RSS (MB) and VRAM (MB if available), used for quick sampling."""
    try:
        mem = human_mb(process.memory_info().rss)
    except Exception:
        mem = 0.0
    vram = 0.0
    if get_vram_fn is not None:
        try:
            vram = get_vram_fn()
        except Exception:
            vram = 0.0
    return mem, vram

# ---------------------------------------------------------------------
# ----------------------  QuantModel loader (ONNX)  -------------------
# ---------------------------------------------------------------------
class QuantModelONNX:
    """
    Лёгкий загрузчик ONNX -> tokenizer + InferenceSession.
    Поддерживает source: 'gdrive' (Google Drive id), 'hf' (HF repo), 'local' (path to dir or .onnx).
    """
    def __init__(self, model_ref: str, source: str = "local", workdir: Optional[str] = None, tokenizer_name: Optional[str] = None, force_download: bool = False):
        self.model_ref = model_ref
        self.source = source
        self.force_download = force_download
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="quant_onnx_"))
        self.tokenizer_name = tokenizer_name
        self.model_path = None
        self.session = None
        self.tokenizer = None
        self._ensure_model()
        self._load_session_and_tokenizer()

    def _ensure_model(self):
        self.workdir.mkdir(parents=True, exist_ok=True)
        onnx_files = list(self.workdir.rglob("*.onnx"))
        if onnx_files and not self.force_download:
            self.model_path = onnx_files[0]
            return

        # Need to download/extract
        if self.source == "gdrive":
            # model_ref expected to be Google Drive file id
            zip_path = str(self.workdir / "model.zip")
            url = f"https://drive.google.com/uc?id={self.model_ref}"
            gdown.download(url, zip_path, quiet=False)
            # try extract if zip
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.workdir)
                os.remove(zip_path)
            except zipfile.BadZipFile:
                # maybe it's just an onnx file renamed
                pass
        elif self.source == "hf":
            huggingface_hub.snapshot_download(repo_id=self.model_ref, local_dir=str(self.workdir), local_dir_use_symlinks=False, resume_download=True)
        elif self.source == "local":
            # accept model_ref as file or dir
            p = Path(self.model_ref)
            if p.exists():
                if p.is_file() and p.suffix == ".onnx":
                    # copy/link
                    self.model_path = p
                    return
                elif p.is_dir():
                    # will scan below
                    pass
            else:
                raise FileNotFoundError(f"Local path {self.model_ref} not found")
        else:
            raise ValueError("source must be one of 'gdrive','hf','local'")

        # find onnx
        onnx_files = list(self.workdir.rglob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx files found in {self.workdir}")
        self.model_path = onnx_files[0]

    def _load_session_and_tokenizer(self):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        try:
            # prefer CUDA if available
            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")
        except Exception:
            pass
        self.session = ort.InferenceSession(str(self.model_path), sess_options=so, providers=providers)
        # tokenizer fallback
        if self.tokenizer_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
            except Exception:
                self.tokenizer = None
        if not self.tokenizer:
            # try loading from model dir
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path.parent), use_fast=True)
            except Exception:
                # fallback to a common tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased", use_fast=True)
                except Exception:
                    self.tokenizer = None

    def encode(self, texts: List[str], batch_size: int = 8, normalize: bool = True, mem_sampler=None) -> np.ndarray:
        """
        Encode texts in batches using ONNX session.
        mem_sampler: optional callable to sample memory (should return tuple (ram_mb, vram_mb))
        """
        if isinstance(texts, str):
            texts = [texts]
        all_embs = []
        n = len(texts)
        i = 0
        while i < n:
            batch = texts[i: i + batch_size]
            i += batch_size
            # tokenize
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not loaded for ONNX model.")
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="np")
            ort_inputs = {k: v for k, v in inputs.items()}
            outs = self.session.run(None, ort_inputs)
            emb = outs[0]  # assume first output is embeddings
            # reduce (seq->sent) if needed
            if emb.ndim == 3:
                # masked mean if attention_mask present
                mask = ort_inputs.get("attention_mask", None)
                if mask is not None:
                    mask = mask.astype(np.float32)[..., None]  # (batch, seq, 1)
                    summed = (emb * mask).sum(axis=1)
                    counts = mask.sum(axis=1)
                    counts = np.clip(counts, 1e-6, None)
                    emb = summed / counts
                else:
                    emb = emb.mean(axis=1)
            if normalize:
                norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
                emb = emb / norms
            all_embs.append(emb)
            # sample memory optionally
            if mem_sampler:
                mem_sampler()
        if not all_embs:
            return np.zeros((0, 0), dtype=float)
        return np.vstack(all_embs)

# ---------------------------------------------------------------------
# ----------------------  Metric computations  ------------------------
# ---------------------------------------------------------------------
def compute_spearman(labels: np.ndarray, preds: np.ndarray) -> float:
    if len(labels) < 2:
        return float("nan")
    return float(spearmanr(labels, preds).correlation)

def compute_pearson(labels: np.ndarray, preds: np.ndarray) -> float:
    if len(labels) < 2:
        return float("nan")
    return float(pearsonr(labels, preds)[0])

def compute_accuracy_at_threshold(labels: np.ndarray, preds: np.ndarray, threshold: float) -> float:
    # labels expected to be 0/1 or continuous -> binarize labels at same threshold if continuous
    lab_bin = (labels >= threshold).astype(int)
    pred_bin = (preds >= threshold).astype(int)
    acc = (lab_bin == pred_bin).mean() if len(labels) > 0 else float("nan")
    return float(acc)

def compute_mrr_and_recall_at_k(grouped: Dict[str, List[Tuple[float,int]]], k: int = 5) -> Tuple[float,float]:
    """
    grouped: dict mapping query_id -> list of tuples (score, is_relevant_flag)
    We will sort by score desc, compute reciprocal rank for first relevant, and recall@k
    """
    rr_list = []
    recall_list = []
    for qid, items in grouped.items():
        # items: list of (score, is_rel)
        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        # find first relevant
        rr = 0.0
        found_any = False
        for idx, (_, is_rel) in enumerate(items_sorted):
            if is_rel:
                found_any = True
                rr = 1.0 / (idx + 1)
                break
        rr_list.append(rr)
        # recall@k: fraction of relevant items in top-k divided by total relevant for that query
        total_rel = sum(1 for _, r in items if r)
        if total_rel == 0:
            recall_list.append(0.0)
        else:
            topk_rel = sum(1 for _, r in items_sorted[:k] if r)
            recall_list.append(topk_rel / total_rel)
    mrr = float(np.mean(rr_list)) if rr_list else float("nan")
    recallk = float(np.mean(recall_list)) if recall_list else float("nan")
    return mrr, recallk

# ---------------------------------------------------------------------
# ----------------------  Small runner utilities  ---------------------
# ---------------------------------------------------------------------
def encode_with_measurement(encoder_callable, texts: List[str], batch_size: int = 8, sampler_steps: int = 4):
    """
    Runs encoder_callable(texts_subset) batching and samples memory/VRAM to approximate peak.
    encoder_callable should accept list[str] and return np.ndarray (n, dim).
    Returns embeddings, total_time_s, throughput(texts/s), peak_ram_mb, peak_vram_mb
    """
    proc = psutil.Process()
    peak_ram = 0.0
    peak_vram = 0.0

    t0 = time.perf_counter()
    # we call encoder in chunks and sample memory after each chunk
    embs = encoder_callable(texts)  # ideally encoder handles batching internally
    t1 = time.perf_counter()

    # single sample after run
    try:
        ram_now = human_mb(proc.memory_info().rss)
        peak_ram = max(peak_ram, ram_now)
    except Exception:
        pass
    # try sample VRAM via torch if available
    try:
        import torch
        if torch.cuda.is_available():
            peak_vram = max(peak_vram, human_mb(torch.cuda.max_memory_allocated()))
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    total_time = t1 - t0
    throughput = len(texts) / max(total_time, 1e-12) if len(texts) > 0 else 0.0
    return embs, total_time, throughput, peak_ram, peak_vram

# ---------------------------------------------------------------------
# ----------------------  UI: Inputs  --------------------------------
# ---------------------------------------------------------------------
st.markdown("### 1) Выберите модели")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Оригинальная модель (Original)")
    orig_source = st.radio("Источник Original:", ("HF (SentenceTransformers)", "ONNX (gdrive/hf/local)"), index=0, key="orig_src")
    if orig_source.startswith("HF"):
        orig_hf_id = st.text_input("HF id (Original)", value="deepvk/USER-BGE-M3", key="orig_hf")
        orig_spec = {"type": "hf", "id": orig_hf_id}
    else:
        orig_onx_source = st.selectbox("ONNX источник", ("local", "gdrive", "hf"), index=0, key="orig_onx_src")
        orig_onx_ref = st.text_input("ONNX id/path (Original)", value="", key="orig_onx_ref")
        orig_tokenizer = st.text_input("Tokenizer (optional) for Original ONNX", value="", key="orig_onx_tok")
        orig_spec = {"type": "onnx", "source": orig_onx_source, "ref": orig_onx_ref, "tokenizer": orig_tokenizer}

with col2:
    st.subheader("Квантованная модель (Quantized)")
    quant_source_choice = st.radio("Источник Quantized:", ("ONNX (gdrive/hf/local)", "HF (SentenceTransformers)"), index=0, key="quant_src")
    if quant_source_choice.startswith("ONNX"):
        quant_onx_source = st.selectbox("ONNX источник", ("gdrive", "hf", "local"), index=0, key="quant_onx_src")
        quant_onx_ref = st.text_input("ONNX id/path (Quantized)", value="", key="quant_onx_ref")
        quant_tokenizer = st.text_input("Tokenizer (optional) for Quant ONNX", value="", key="quant_onx_tok")
        quant_spec = {"type": "onnx", "source": quant_onx_source, "ref": quant_onx_ref, "tokenizer": quant_tokenizer}
    else:
        quant_hf_id = st.text_input("HF id (Quantized)", value="", key="quant_hf")
        quant_spec = {"type": "hf", "id": quant_hf_id}

st.markdown("---")
st.markdown("### 2) Данные (поддержка русского датасетов)")
st.markdown(
    "Варианты: загрузить CSV (формат ниже), или указать HF dataset id. "
    "Для простоты вы можете загрузить CSV с парами: `sentence1,sentence2,label` (label: float или 0/1). "
    "Для retrieval/MRR: используйте формат `query_id,query,doc,is_relevant(0/1)`."
)
data_source = st.selectbox("Источник данных:", ("Upload CSV", "HuggingFace dataset (id)"), index=0)

df = None
dataset_mode = "pairwise"  # or "retrieval"
if data_source == "Upload CSV":
    uploaded = st.file_uploader("CSV: (pairwise -> sentence1,sentence2,label) OR (retrieval -> query_id,query,doc,is_relevant)", type=["csv", "txt"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Загружен CSV с {len(df)} строками.")
            st.write(df.head(5))
            # try to detect mode
            if {"query_id", "query", "doc", "is_relevant"}.issubset(set(df.columns)):
                dataset_mode = "retrieval"
            elif {"sentence1", "sentence2", "label"}.issubset(set(df.columns)):
                dataset_mode = "pairwise"
            else:
                # ask user to choose
                dataset_mode = st.selectbox("Выберите режим данных:", ("pairwise", "retrieval"), index=0)
        except Exception as e:
            st.error(f"Не смог загрузить CSV: {e}")
            st.text(traceback.format_exc())
elif data_source == "HuggingFace dataset (id)":
    hf_id = st.text_input("HF dataset id (например, user/dataset)", value="", key="hf_dataset")
    load_preview = st.button("Загрузить HF dataset preview")
    if load_preview and hf_id:
        # we will try to load via datasets library - but keep it optional
        try:
            from datasets import load_dataset
            dd = load_dataset(hf_id, split="train[:200]")
            df = pd.DataFrame(dd[:200])
            st.success(f"Загружено {len(df)} примеров (preview 200).")
            st.write(df.head())
            # try to detect
            if {"sentence1", "sentence2", "label"}.issubset(set(df.columns)):
                dataset_mode = "pairwise"
            else:
                dataset_mode = st.selectbox("Выберите режим данных:", ("pairwise", "retrieval"), index=0)
        except Exception as e:
            st.error(f"Ошибка при загрузке HF dataset: {e}")
            st.text(traceback.format_exc())

# basic params
max_samples = st.slider("Макс. примеров для теста (чтобы не перегружать)", 1, 2048, 256)
batch_size = st.slider("Batch size (внутри encode)", 1, 256, 8)
threshold = st.slider("Порог для Accuracy@Threshold", 0.0, 1.0, 0.7)
recall_k = st.number_input("K для Recall@K", min_value=1, max_value=100, value=5)

st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    if st.button("♻️ Сбросить сессию/кэши"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.experimental_rerun()
with col_b:
    run_button = st.button("🚀 Запустить тест")

# ---------------------------------------------------------------------
# ----------------------  Run logic  ---------------------------------
# ---------------------------------------------------------------------
if run_button:
    # Basic checks
    try:
        if df is None:
            st.warning("Загрузите датасет (CSV) или HF dataset preview прежде чем запускать.")
            st.stop()

        # trim dataset to max_samples intelligently depending on mode
        if dataset_mode == "pairwise":
            if not {"sentence1", "sentence2", "label"}.issubset(set(df.columns)):
                st.error("Для pairwise необходимы колонки: sentence1,sentence2,label")
                st.stop()
            df_use = df[["sentence1", "sentence2", "label"]].dropna().reset_index(drop=True).iloc[:max_samples]
            sentences_a = df_use["sentence1"].astype(str).tolist()
            sentences_b = df_use["sentence2"].astype(str).tolist()
            labels = df_use["label"].astype(float).to_numpy()
            is_retrieval = False
        else:
            # retrieval mode expects query_id, query, doc, is_relevant
            if not {"query_id", "query", "doc", "is_relevant"}.issubset(set(df.columns)):
                st.error("Для retrieval необходимы колонки: query_id, query, doc, is_relevant")
                st.stop()
            df_use = df[["query_id", "query", "doc", "is_relevant"]].dropna().reset_index(drop=True).iloc[:max_samples]
            is_retrieval = True

        # Load models
        st.info("Загрузка моделей...")
        # Original loader
        orig_encoder = None
        orig_name = "Original"
        if orig_spec["type"] == "hf":
            st.write("Загружаю Original HF model (SentenceTransformer)...")
            orig_model = SentenceTransformer(orig_spec["id"])
            def orig_encode(texts: List[str]):
                return orig_model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size)
            orig_encoder = orig_encode
            orig_name = orig_spec["id"]
        else:
            st.write("Загружаю Original ONNX...")
            om = QuantModelONNX(model_ref=orig_spec["ref"], source=orig_spec.get("source", "local"), tokenizer_name=orig_spec.get("tokenizer") or None)
            def orig_encode(texts: List[str]):
                return om.encode(texts, batch_size=batch_size, normalize=True)
            orig_encoder = orig_encode
            orig_name = f"ONNX:{Path(om.model_path).name}"

        # Quant loader
        quant_encoder = None
        quant_name = "Quant"
        if quant_spec["type"] == "hf":
            if not quant_spec["id"]:
                st.error("Выберите HF id для quantized модели или выберите ONNX.")
                st.stop()
            st.write("Загружаю Quant (HF SentenceTransformer)...")
            q_model = SentenceTransformer(quant_spec["id"])
            def q_encode(texts: List[str]):
                return q_model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size)
            quant_encoder = q_encode
            quant_name = quant_spec["id"]
        else:
            st.write("Загружаю Quant (ONNX)...")
            qm = QuantModelONNX(model_ref=quant_spec["ref"], source=quant_spec.get("source", "local"), tokenizer_name=quant_spec.get("tokenizer") or None)
            def q_encode(texts: List[str]):
                return qm.encode(texts, batch_size=batch_size, normalize=True)
            quant_encoder = q_encode
            quant_name = f"ONNX:{Path(qm.model_path).name}"

        # Prepare memory sampler function
        proc = psutil.Process()
        try:
            import torch
            def get_vram():
                if torch.cuda.is_available():
                    return human_mb(torch.cuda.max_memory_allocated())
                return 0.0
        except Exception:
            def get_vram():
                return 0.0

        # Encoding + measurement
        st.info("Запускаю inference и замеряю метрики...")
        progress = st.progress(0)

        if not is_retrieval:
            # Pairwise: encode both lists and compute per-pair cosine
            n = len(sentences_a)
            progress_step = max(1, n // 10)
            # Original
            s0 = time.perf_counter()
            emb_o, t_o, thr_o, ram_o, vram_o = encode_with_measurement(lambda txts: orig_encoder(txts), sentences_a + sentences_b, batch_size)
            # emb_o contains embeddings for all texts (we encoded concatenated list)
            # split
            emb_a_o = emb_o[:n]
            emb_b_o = emb_o[n: n*2]
            # Quant
            emb_q_all, t_q, thr_q, ram_q, vram_q = encode_with_measurement(lambda txts: quant_encoder(txts), sentences_a + sentences_b, batch_size)
            emb_a_q = emb_q_all[:n]
            emb_b_q = emb_q_all[n: n*2]

            # compute cosine similarities per pair
            def cos_sim_batch(A, B):
                A = np.asarray(A)
                B = np.asarray(B)
                # align dims
                if A.shape[1] != B.shape[1]:
                    m = min(A.shape[1], B.shape[1])
                    A = A[:, :m]; B = B[:, :m]
                A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
                B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
                return (A * B).sum(axis=1)

            scores_orig = cos_sim_batch(emb_a_o, emb_b_o)
            scores_quant = cos_sim_batch(emb_a_q, emb_b_q)

            # Metrics
            spearman_o = compute_spearman(labels, scores_orig)
            pearson_o = compute_pearson(labels, scores_orig)
            acc_o = compute_accuracy_at_threshold(labels, scores_orig, threshold)

            spearman_q = compute_spearman(labels, scores_quant)
            pearson_q = compute_pearson(labels, scores_quant)
            acc_q = compute_accuracy_at_threshold(labels, scores_quant, threshold)

            # Prepare results dataframe
            df_results = pd.DataFrame([
                {
                    "Model": orig_name,
                    "Latency(s)": float(t_o),
                    "Throughput(texts/s)": float(thr_o),
                    "Peak RAM (MB)": float(ram_o),
                    "Peak VRAM (MB)": float(vram_o),
                    "Spearman": spearman_o,
                    "Pearson": pearson_o,
                    f"Accuracy@{threshold:.2f}": acc_o
                },
                {
                    "Model": quant_name,
                    "Latency(s)": float(t_q),
                    "Throughput(texts/s)": float(thr_q),
                    "Peak RAM (MB)": float(ram_q),
                    "Peak VRAM (MB)": float(vram_q),
                    "Spearman": spearman_q,
                    "Pearson": pearson_q,
                    f"Accuracy@{threshold:.2f}": acc_q
                }
            ])

            st.subheader("📊 Результаты (pairwise)")
            st.dataframe(df_results)

            # show small distributions
            st.subheader("🎯 Качество: распределения прогнозов")
            st.write("Original (первые 10 прогнозов):")
            st.write(scores_orig[:10])
            st.write("Quantized (первые 10 прогнозов):")
            st.write(scores_quant[:10])

            # CSV download
            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Скачать CSV с результатами", csv, file_name="model_comparison_pairwise.csv", mime="text/csv")

        else:
            # Retrieval mode: we will compute embeddings for queries and docs, group by query_id and compute MRR/Recall@K
            # build groups
            df_group = df_use.copy()
            # unique queries
            queries = df_group[["query_id", "query"]].drop_duplicates(subset=["query_id"]).reset_index(drop=True)
            docs = df_group[["query_id", "doc", "is_relevant"]].reset_index(drop=True)

            # encode all queries and docs
            query_texts = queries["query"].astype(str).tolist()
            doc_texts = docs["doc"].astype(str).tolist()

            # encode queries
            emb_qry_orig, t_qo, thr_qo, ram_qo, vram_qo = encode_with_measurement(lambda txts: orig_encoder(txts), query_texts, batch_size)
            emb_docs_orig, t_do, thr_do, ram_do, vram_do = encode_with_measurement(lambda txts: orig_encoder(txts), doc_texts, batch_size)

            emb_qry_quant, t_qq, thr_qq, ram_qq, vram_qq = encode_with_measurement(lambda txts: quant_encoder(txts), query_texts, batch_size)
            emb_docs_quant, t_dq, thr_dq, ram_dq, vram_dq = encode_with_measurement(lambda txts: quant_encoder(txts), doc_texts, batch_size)

            # now compute per-query scores and group
            # map doc embeddings index to query
            grouped_orig = {}
            grouped_quant = {}
            # prepare mapping: for each docs row, compute cosine with the corresponding query
            # build index of query to its index in query_texts
            qid_to_idx = {qid: idx for idx, qid in enumerate(queries["query_id"].tolist())}
            # docs order corresponds to doc_texts order
            doc_idx = 0
            for _, row in docs.iterrows():
                qid = row["query_id"]
                qidx = qid_to_idx[qid]
                # compute cosine between emb_qry_orig[qidx] and emb_docs_orig[doc_idx]
                s_orig = float(np.dot(emb_qry_orig[qidx], emb_docs_orig[doc_idx]) / ((np.linalg.norm(emb_qry_orig[qidx]) * np.linalg.norm(emb_docs_orig[doc_idx])) + 1e-12))
                s_quant = float(np.dot(emb_qry_quant[qidx], emb_docs_quant[doc_idx]) / ((np.linalg.norm(emb_qry_quant[qidx]) * np.linalg.norm(emb_docs_quant[doc_idx])) + 1e-12))
                is_rel = int(row["is_relevant"])
                grouped_orig.setdefault(qid, []).append((s_orig, is_rel))
                grouped_quant.setdefault(qid, []).append((s_quant, is_rel))
                doc_idx += 1

            mrr_o, recallk_o = compute_mrr_and_recall_at_k(grouped_orig, k=recall_k)
            mrr_q, recallk_q = compute_mrr_and_recall_at_k(grouped_quant, k=recall_k)

            # resource totals approximate (sum query+doc times)
            res_orig_time = float(t_qo + t_do)
            res_quant_time = float(t_qq + t_dq)
            res_orig_ram = float(max(ram_qo, ram_do))
            res_quant_ram = float(max(ram_qq, ram_dq))
            res_orig_vram = float(max(vram_qo, vram_do))
            res_quant_vram = float(max(vram_qq, vram_dq))

            df_results = pd.DataFrame([
                {
                    "Model": orig_name,
                    "Latency(s)_approx": res_orig_time,
                    "Throughput(texts/s)_approx": float(len(query_texts) / max(res_orig_time, 1e-12)),
                    "Peak RAM (MB)": res_orig_ram,
                    "Peak VRAM (MB)": res_orig_vram,
                    "MRR": mrr_o,
                    f"Recall@{recall_k}": recallk_o
                },
                {
                    "Model": quant_name,
                    "Latency(s)_approx": res_quant_time,
                    "Throughput(texts/s)_approx": float(len(query_texts) / max(res_quant_time, 1e-12)),
                    "Peak RAM (MB)": res_quant_ram,
                    "Peak VRAM (MB)": res_quant_vram,
                    "MRR": mrr_q,
                    f"Recall@{recall_k}": recallk_q
                }
            ])
            st.subheader("📊 Результаты (retrieval)")
            st.dataframe(df_results)
            csv = df_results.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Скачать CSV с результатами (retrieval)", csv, file_name="model_comparison_retrieval.csv", mime="text/csv")

        st.success("Готово ✅")
    except Exception as e:
        st.error(f"Ошибка во время теста: {e}")
        st.text(traceback.format_exc())
