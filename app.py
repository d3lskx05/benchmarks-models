# compare_models_app.py
import os
import time
import psutil
import torch
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from transformers import AutoTokenizer
from scipy.stats import spearmanr, pearsonr

st.set_page_config(page_title="Model Benchmark", layout="wide")

# ============================================================
# 🔧 Metrics
# ============================================================
def cosine_similarity_matrix(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Косинусное сходство между двумя наборами векторов."""
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-12)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-12)
    return np.matmul(emb1_norm, emb2_norm.T)

def accuracy_at_threshold(sim: np.ndarray, labels: np.ndarray, threshold: float = 0.5):
    preds = (sim >= threshold).astype(int)
    labels_bin = (labels >= 0.5).astype(int)
    return (preds == labels_bin).mean()

def mean_reciprocal_rank(sim_matrix: np.ndarray) -> float:
    """Mean Reciprocal Rank (MRR)."""
    ranks = []
    for i in range(len(sim_matrix)):
        row = sim_matrix[i]
        sorted_idx = np.argsort(-row)
        rank = np.where(sorted_idx == i)[0][0] + 1
        ranks.append(1 / rank)
    return float(np.mean(ranks))

def recall_at_k(sim_matrix: np.ndarray, k: int = 5) -> float:
    """Recall@K."""
    hits = 0
    for i in range(len(sim_matrix)):
        row = sim_matrix[i]
        top_k = np.argsort(-row)[:k]
        if i in top_k:
            hits += 1
    return hits / len(sim_matrix)

def measure_latency_and_memory(model, sentences: List[str], batch_size: int = 16) -> Tuple[float, float]:
    """Измерение latency и RAM."""
    proc = psutil.Process()
    t0 = time.perf_counter()
    _ = model.encode(sentences, batch_size=batch_size, convert_to_numpy=True)
    t1 = time.perf_counter()
    mem = proc.memory_info().rss / 1024**2
    return t1 - t0, mem


# ============================================================
# 🎛️ UI
# ============================================================
st.title("🔍 Benchmark разных моделей (русский датасет)")
st.markdown("Сравниваем **любые две модели** по качеству и ресурсам.")

col1, col2 = st.columns(2)
with col1:
    model_a_id = st.text_input("Модель A", "intfloat/multilingual-e5-base")
with col2:
    model_b_id = st.text_input("Модель B", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

dataset_name = st.selectbox("Выберите датасет", ["ai-forever/ruSTS", "paraphrase_mining"], index=0)
n_samples = st.slider("Количество примеров", 10, 500, 50, step=10)
batch_size = st.slider("Batch size для инференса", 1, 64, 16)
recall_k = st.slider("Recall@K", 1, 10, 5)

run_button = st.button("🚀 Запустить сравнение")
reset_button = st.button("♻️ Сбросить сессию")

if reset_button:
    st.cache_resource.clear()
    st.cache_data.clear()
    st.experimental_rerun()

# ============================================================
# 🚀 Benchmark
# ============================================================
if run_button:
    try:
        # === Load dataset ===
        st.write(f"📥 Загружаем датасет: {dataset_name}")
        ds = load_dataset(dataset_name, split="test")
        if dataset_name == "ai-forever/ruSTS":
            pairs = [(s1, s2, float(score)) for s1, s2, score in zip(ds["sentence1"], ds["sentence2"], ds["score"])]
        else:
            pairs = [(s, t, float(label)) for s, t, label in zip(ds["sentence1"], ds["sentence2"], ds["label"])]

        pairs = pairs[:n_samples]
        sents1, sents2, labels = zip(*pairs)
        labels = np.array(labels)

        # === Load models ===
        st.write(f"🚀 Загружаем модель A: {model_a_id}")
        model_a = SentenceTransformer(model_a_id)
        st.write(f"🚀 Загружаем модель B: {model_b_id}")
        model_b = SentenceTransformer(model_b_id)

        # === Encode ===
        st.write("🔢 Считаем эмбеддинги...")
        emb_a1 = model_a.encode(list(sents1), batch_size=batch_size, convert_to_numpy=True)
        emb_a2 = model_a.encode(list(sents2), batch_size=batch_size, convert_to_numpy=True)
        emb_b1 = model_b.encode(list(sents1), batch_size=batch_size, convert_to_numpy=True)
        emb_b2 = model_b.encode(list(sents2), batch_size=batch_size, convert_to_numpy=True)

        # === Metrics ===
        sim_a = np.array([np.dot(u / np.linalg.norm(u), v / np.linalg.norm(v)) for u, v in zip(emb_a1, emb_a2)])
        sim_b = np.array([np.dot(u / np.linalg.norm(u), v / np.linalg.norm(v)) for u, v in zip(emb_b1, emb_b2)])

        spearman_a = spearmanr(sim_a, labels).correlation
        pearson_a = pearsonr(sim_a, labels)[0]
        spearman_b = spearmanr(sim_b, labels).correlation
        pearson_b = pearsonr(sim_b, labels)[0]

        sim_matrix_a = cosine_similarity_matrix(emb_a1, emb_a2)
        sim_matrix_b = cosine_similarity_matrix(emb_b1, emb_b2)

        acc_a = accuracy_at_threshold(sim_a, labels)
        acc_b = accuracy_at_threshold(sim_b, labels)
        mrr_a = mean_reciprocal_rank(sim_matrix_a)
        mrr_b = mean_reciprocal_rank(sim_matrix_b)
        r_a = recall_at_k(sim_matrix_a, recall_k)
        r_b = recall_at_k(sim_matrix_b, recall_k)

        latency_a, mem_a = measure_latency_and_memory(model_a, list(sents1))
        latency_b, mem_b = measure_latency_and_memory(model_b, list(sents1))

        # === Table ===
        data = {
            "Model": ["A", "B"],
            "Spearman": [spearman_a, spearman_b],
            "Pearson": [pearson_a, pearson_b],
            "Accuracy@0.5": [acc_a, acc_b],
            "MRR": [mrr_a, mrr_b],
            f"Recall@{recall_k}": [r_a, r_b],
            "Latency (s)": [latency_a, latency_b],
            "Memory (MB)": [mem_a, mem_b],
        }
        df = pd.DataFrame(data)
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Скачать результаты (CSV)",
            csv,
            "benchmark_results.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Ошибка: {e}")
