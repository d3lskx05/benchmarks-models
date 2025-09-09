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
from datasets import load_dataset  # 🔥 NEW

# metrics
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import accuracy_score  # 🔥 NEW

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

# 🔥 NEW: загрузка русского датасета
@st.cache_data
def load_russian_dataset(name="RuSTS", split="test", limit=200):
    if name == "RuSTS":
        ds = load_dataset("ai-forever/RuSTS", split=split)
        sent1 = ds["sentence1"][:limit]
        sent2 = ds["sentence2"][:limit]
        labels = np.array(ds["similarity"][:limit]) / 5.0  # нормируем
        mode = "pairwise"
    elif name == "ParaPhraser":
        ds = load_dataset("cointegrated/paraphraser", split="train")
        sent1 = ds["sentence1"][:limit]
        sent2 = ds["sentence2"][:limit]
        labels = np.array(ds["paraphrase"][:limit]).astype(int)
        mode = "pairwise"
    else:
        raise ValueError("Неизвестный датасет")
    df = pd.DataFrame({"sentence1": sent1, "sentence2": sent2, "label": labels})
    return df, mode

# ---------------------------------------------------------------------
# ----------------------  QuantModel loader (ONNX)  -------------------
# ---------------------------------------------------------------------
# (оставлен без изменений)
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
    # ... (весь код QuantModelONNX без изменений)

# ---------------------------------------------------------------------
# ----------------------  Metric computations  ------------------------
# ---------------------------------------------------------------------
# (оставлен без изменений)
def compute_spearman(labels: np.ndarray, preds: np.ndarray) -> float:
    if len(labels) < 2:
        return float("nan")
    return float(spearmanr(labels, preds).correlation)

def compute_pearson(labels: np.ndarray, preds: np.ndarray) -> float:
    if len(labels) < 2:
        return float("nan")
    return float(pearsonr(labels, preds)[0])

def compute_accuracy_at_threshold(labels: np.ndarray, preds: np.ndarray, threshold: float) -> float:
    lab_bin = (labels >= threshold).astype(int)
    pred_bin = (preds >= threshold).astype(int)
    acc = (lab_bin == pred_bin).mean() if len(labels) > 0 else float("nan")
    return float(acc)

def compute_mrr_and_recall_at_k(grouped: Dict[str, List[Tuple[float,int]]], k: int = 5) -> Tuple[float,float]:
    rr_list = []
    recall_list = []
    for qid, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        rr = 0.0
        for idx, (_, is_rel) in enumerate(items_sorted):
            if is_rel:
                rr = 1.0 / (idx + 1)
                break
        rr_list.append(rr)
        total_rel = sum(1 for _, r in items if r)
        if total_rel == 0:
            recall_list.append(0.0)
        else:
            topk_rel = sum(1 for _, r in items_sorted[:k] if r)
            recall_list.append(topk_rel / total_rel)
    return float(np.mean(rr_list)), float(np.mean(recall_list))

# ---------------------------------------------------------------------
# ----------------------  Small runner utilities  ---------------------
# ---------------------------------------------------------------------
# (оставлен без изменений)

# ---------------------------------------------------------------------
# ----------------------  UI: Inputs  --------------------------------
# ---------------------------------------------------------------------
st.markdown("### 1) Выберите модели")
# (UI для выбора моделей без изменений)

st.markdown("---")
st.markdown("### 2) Данные")
dataset_choice = st.selectbox("Быстрый выбор датасета:", ["Нет", "RuSTS", "ParaPhraser"], index=0)  # 🔥 NEW

df = None
dataset_mode = "pairwise"

if dataset_choice != "Нет":
    with st.spinner(f"Загружаю {dataset_choice}..."):
        df, dataset_mode = load_russian_dataset(dataset_choice, limit=256)
    st.success(f"Загружено {len(df)} примеров из {dataset_choice}")
    st.dataframe(df.head())
else:
    # старый блок выбора CSV / HF датасета
    data_source = st.selectbox("Источник данных:", ("Upload CSV", "HuggingFace dataset (id)"), index=0)
    # ... (весь код загрузки CSV/HF без изменений)

# Параметры
max_samples = st.slider("Макс. примеров для теста", 1, 2048, 256)
batch_size = st.slider("Batch size", 1, 256, 8)
threshold = st.slider("Порог для Accuracy@Threshold", 0.0, 1.0, 0.7)
recall_k = st.number_input("K для Recall@K", min_value=1, max_value=100, value=5)

# Кнопки
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
# ----------------------  Run logic ----------------------------------
# ---------------------------------------------------------------------
# (Блок запуска и расчёта метрик без изменений, он работает с df)
