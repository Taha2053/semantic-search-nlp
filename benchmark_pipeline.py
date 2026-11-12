"""
Scientific Benchmark Pipeline for Semantic Search vs Lexical Retrieval
Evaluates TF-IDF, BM25, SBERT Dense Bi-Encoder, and Hybrid Retrieval (RRF & Convex Blending)
on the NFCorpus BEIR Benchmark.
"""

import json
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoTokenizer, AutoModel

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = "data/nfcorpus"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("=== 1. LOADING NFCORPUS DATASET ===")

# Load Corpus
corpus = {}
with open(os.path.join(DATA_DIR, "corpus.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        cid = item["_id"]
        title = item.get("title", "")
        text = item.get("text", "")
        # Combine title and text with structured format
        full_text = f"{title}. {text}".strip()
        corpus[cid] = {
            "title": title,
            "text": text,
            "full_text": full_text
        }

print(f"Total documents in corpus: {len(corpus):,}")

# Load Queries
queries = {}
with open(os.path.join(DATA_DIR, "queries.jsonl"), "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        queries[item["_id"]] = item["text"]

print(f"Total queries available: {len(queries):,}")

# Load Qrels (Relevance Judgments) - Test Split
qrels_test = {}
with open(os.path.join(DATA_DIR, "qrels", "test.tsv"), "r", encoding="utf-8") as f:
    header = f.readline()  # skip query-id \t corpus-id \t score
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            qid, cid, score = parts[0], parts[1], int(parts[2])
            if qid not in qrels_test:
                qrels_test[qid] = {}
            qrels_test[qid][cid] = score

# Keep test queries that have gold relevance judgements
eval_qids = [qid for qid in queries if qid in qrels_test and len(qrels_test[qid]) > 0]
print(f"Evaluated test queries with qrels: {len(eval_qids)}")

# Convert corpus to list for indexing
doc_ids = list(corpus.keys())
doc_texts = [corpus[cid]["full_text"] for cid in doc_ids]

print("\n=== 2. CORPUS PREPROCESSING & EXPLORATION ===")
# Exploration statistics
doc_lens = [len(t.split()) for t in doc_texts]
query_lens = [len(queries[qid].split()) for qid in eval_qids]

corpus_stats = {
    "num_documents": len(corpus),
    "num_test_queries": len(eval_qids),
    "doc_len_mean": float(np.mean(doc_lens)),
    "doc_len_median": float(np.median(doc_lens)),
    "doc_len_max": int(np.max(doc_lens)),
    "doc_len_min": int(np.min(doc_lens)),
    "query_len_mean": float(np.mean(query_lens)),
    "query_len_median": float(np.median(query_lens)),
    "total_qrels_pairs": sum(len(qrels) for qrels in qrels_test.values())
}
print("Corpus Statistics:", json.dumps(corpus_stats, indent=2))

def tokenize_lexical(text):
    """Lowercased alphanumeric tokenization with length >= 2."""
    tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
    return tokens

tokenized_corpus = [tokenize_lexical(t) for t in doc_texts]

print("\n=== 3. BUILDING RETRIEVAL MODELS ===")

# Model 1: TF-IDF
print("Building TF-IDF Vectorizer...")
t0 = time.time()
tfidf_vectorizer = TfidfVectorizer(
    tokenizer=tokenize_lexical,
    lowercase=True,
    stop_words="english",
    max_features=50000,
    sublinear_tf=True
)
tfidf_matrix = tfidf_vectorizer.fit_transform(doc_texts)
tfidf_index_time = time.time() - t0
print(f"TF-IDF indexed in {tfidf_index_time:.2f}s, shape: {tfidf_matrix.shape}")

# Model 2: BM25
print("Building BM25Okapi Index...")
t0 = time.time()
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
bm25_index_time = time.time() - t0
print(f"BM25 indexed in {bm25_index_time:.2f}s")

# Model 3: SBERT Dense Bi-Encoder
print("Loading and encoding with Sentence-BERT (all-MiniLM-L6-v2) on CUDA...")
model_path = "/home/taha/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
tokenizer = AutoTokenizer.from_pretrained(model_path)
dense_model = AutoModel.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
dense_model = dense_model.to(device)
dense_model.eval()

def encode_texts(texts, batch_size=64):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dense_model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            all_embeddings.append(normalized.cpu())
    return torch.cat(all_embeddings, dim=0)

t0 = time.time()
doc_embeddings = encode_texts(doc_texts, batch_size=128)
dense_index_time = time.time() - t0
print(f"Dense embeddings encoded in {dense_index_time:.2f}s, shape: {doc_embeddings.shape}")

# Store doc embeddings on GPU for fast retrieval
doc_embeddings_gpu = doc_embeddings.to(device)

print("\n=== 4. EVALUATION PROTOCOL ===")
def compute_ir_metrics(retrieved_ids, gold_qrels, k_vals=[1, 3, 5, 10, 20]):
    """
    Computes standard IR metrics:
    - MRR@10
    - NDCG@10 (graded relevance: 2=high, 1=marginal, 0=non-relevant)
    - MAP@10
    - Precision@k
    - Recall@k
    """
    metrics = {}
    
    # Binarized relevance (score >= 1)
    relevant_cids = {cid: score for cid, score in gold_qrels.items() if score >= 1}
    total_relevant = len(relevant_cids)
    if total_relevant == 0:
        return None
    
    # 1. MRR@10
    mrr_10 = 0.0
    for rank, cid in enumerate(retrieved_ids[:10], start=1):
        if cid in relevant_cids:
            mrr_10 = 1.0 / rank
            break
    metrics["MRR@10"] = mrr_10
    
    # 2. NDCG@10
    def dcg(scores):
        return sum((2**s - 1) / np.log2(idx + 2) for idx, s in enumerate(scores))
    
    actual_scores = [gold_qrels.get(cid, 0) for cid in retrieved_ids[:10]]
    dcg_val = dcg(actual_scores)
    ideal_scores = sorted(gold_qrels.values(), reverse=True)[:10]
    idcg_val = dcg(ideal_scores)
    metrics["NDCG@10"] = (dcg_val / idcg_val) if idcg_val > 0 else 0.0
    
    # 3. MAP@10
    num_hits = 0
    sum_prec = 0.0
    for rank, cid in enumerate(retrieved_ids[:10], start=1):
        if cid in relevant_cids:
            num_hits += 1
            sum_prec += num_hits / rank
    metrics["MAP@10"] = sum_prec / min(total_relevant, 10) if total_relevant > 0 else 0.0
    
    # 4. Precision & Recall @ k
    for k in k_vals:
        top_k = retrieved_ids[:k]
        hits_k = sum(1 for cid in top_k if cid in relevant_cids)
        metrics[f"P@{k}"] = hits_k / k
        metrics[f"R@{k}"] = hits_k / total_relevant if total_relevant > 0 else 0.0
        
    return metrics

# Search functions
def search_tfidf(query_text, top_k=100):
    q_vec = tfidf_vectorizer.transform([query_text])
    sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(-sims)[:top_k]
    return [(doc_ids[idx], float(sims[idx])) for idx in top_indices]

def search_bm25(query_text, top_k=100):
    q_tokens = tokenize_lexical(query_text)
    scores = bm25.get_scores(q_tokens)
    top_indices = np.argsort(-scores)[:top_k]
    return [(doc_ids[idx], float(scores[idx])) for idx in top_indices]

def search_dense(query_text, top_k=100):
    inputs = tokenizer([query_text], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dense_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        mean_pooled = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        q_emb = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        # Cosine similarity on GPU
        sims = torch.mm(q_emb, doc_embeddings_gpu.t()).squeeze(0)
        vals, indices = torch.topk(sims, k=top_k)
        return [(doc_ids[idx.item()], val.item()) for idx, val in zip(indices, vals)]

def search_rrf(bm25_res, dense_res, k=60, top_k=100):
    """Reciprocal Rank Fusion (Cormack et al., SIGIR 2009)"""
    scores = {}
    for rank, (cid, _) in enumerate(bm25_res, start=1):
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
    for rank, (cid, _) in enumerate(dense_res, start=1):
        scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank))
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_docs

def search_hybrid_convex(bm25_res, dense_res, alpha=0.5, top_k=100):
    """Normalized score convex combination: alpha * dense + (1-alpha) * bm25"""
    bm25_dict = dict(bm25_res)
    dense_dict = dict(dense_res)
    
    # Min-max normalize scores
    def norm_dict(d):
        if not d:
            return {}
        vals = list(d.values())
        min_v, max_v = min(vals), max(vals)
        if max_v == min_v:
            return {k: 1.0 for k in d}
        return {k: (v - min_v) / (max_v - min_v) for k, v in d.items()}
    norm_bm25 = norm_dict(bm25_dict)
    norm_dense = norm_dict(dense_dict)
    all_cids = set(norm_bm25.keys()) | set(norm_dense.keys())
    combined = {}
    for cid in all_cids:
        score_b = norm_bm25.get(cid, 0.0)
        score_d = norm_dense.get(cid, 0.0)
        combined[cid] = alpha * score_d + (1.0 - alpha) * score_b
    sorted_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_docs
model_names = [
    "TF-IDF (Cosine)",
    "BM25 (Okapi)",
    "SBERT (all-MiniLM-L6-v2)",
    "Hybrid (RRF, k=60)",
    "Hybrid (Convex, α=0.5)"
]

results = {name: [] for name in model_names}
latencies = {name: [] for name in model_names}
detailed_per_query = []

for q_idx, qid in enumerate(eval_qids):
    q_text = queries[qid]
    gold = qrels_test[qid]
    
    # 1. TF-IDF
    t0 = time.perf_counter()
    res_tfidf = search_tfidf(q_text, top_k=100)
    lat_tfidf = (time.perf_counter() - t0) * 1000.0
    latencies["TF-IDF (Cosine)"].append(lat_tfidf)
    m_tfidf = compute_ir_metrics([doc for doc, _ in res_tfidf], gold)
    results["TF-IDF (Cosine)"].append(m_tfidf)
    
    # 2. BM25
    t0 = time.perf_counter()
    res_bm25 = search_bm25(q_text, top_k=100)
    lat_bm25 = (time.perf_counter() - t0) * 1000.0
    latencies["BM25 (Okapi)"].append(lat_bm25)
    m_bm25 = compute_ir_metrics([doc for doc, _ in res_bm25], gold)
    results["BM25 (Okapi)"].append(m_bm25)
    
    # 3. Dense SBERT
    t0 = time.perf_counter()
    res_dense = search_dense(q_text, top_k=100)
    lat_dense = (time.perf_counter() - t0) * 1000.0
    latencies["SBERT (all-MiniLM-L6-v2)"].append(lat_dense)
    m_dense = compute_ir_metrics([doc for doc, _ in res_dense], gold)
    results["SBERT (all-MiniLM-L6-v2)"].append(m_dense)
    
    # 4. Hybrid RRF
    t0 = time.perf_counter()
    res_rrf = search_rrf(res_bm25, res_dense, k=60, top_k=100)
    lat_rrf = lat_bm25 + lat_dense + (time.perf_counter() - t0) * 1000.0
    latencies["Hybrid (RRF, k=60)"].append(lat_rrf)
    m_rrf = compute_ir_metrics([doc for doc, _ in res_rrf], gold)
    results["Hybrid (RRF, k=60)"].append(m_rrf)
    
    # 5. Hybrid Convex
    t0 = time.perf_counter()
    res_convex = search_hybrid_convex(res_bm25, res_dense, alpha=0.5, top_k=100)
    lat_convex = lat_bm25 + lat_dense + (time.perf_counter() - t0) * 1000.0
    latencies["Hybrid (Convex, α=0.5)"].append(lat_convex)
    m_convex = compute_ir_metrics([doc for doc, _ in res_convex], gold)
    results["Hybrid (Convex, α=0.5)"].append(m_convex)
    
    detailed_per_query.append({
        "qid": qid,
        "query": q_text,
        "num_gold": len(gold),
        "ndcg10_bm25": m_bm25["NDCG@10"],
        "ndcg10_dense": m_dense["NDCG@10"],
        "ndcg10_hybrid": m_rrf["NDCG@10"],
        "mrr10_bm25": m_bm25["MRR@10"],
        "mrr10_dense": m_dense["MRR@10"],
        "mrr10_hybrid": m_rrf["MRR@10"],
        "bm25_top3": [cid for cid, _ in res_bm25[:3]],
        "dense_top3": [cid for cid, _ in res_dense[:3]],
        "gold_cids": list(gold.keys())
    })

# Aggregate metrics
summary_rows = []
metric_keys = ["NDCG@10", "MRR@10", "MAP@10", "P@5", "P@10", "R@5", "R@10", "R@20"]

for name in model_names:
    row = {"Model": name}
    for m in metric_keys:
        vals = [res[m] for res in results[name] if res is not None]
        row[m] = float(np.mean(vals))
    row["Latency (ms)"] = float(np.mean(latencies[name]))
    row["Throughput (QPS)"] = float(1000.0 / np.mean(latencies[name]))
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print("\n=== BENCHMARK SUMMARY TABLE ===")
print(summary_df.to_string(index=False))

summary_df.to_csv("results/benchmark_summary.csv", index=False)
with open("results/per_query_results.json", "w", encoding="utf-8") as f:
    json.dump(detailed_per_query, f, indent=2)

print("\n=== 6. GENERATING COMPARATIVE FIGURES ===")

# Plot 1: IR Metrics Comparison Barplot
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
x = np.arange(len(metric_keys))
width = 0.16
colors = ["#4A5568", "#2B6CB0", "#319795", "#805AD5", "#DD6B20"]

for i, name in enumerate(model_names):
    vals = [summary_df.loc[summary_df["Model"] == name, m].values[0] for m in metric_keys]
    ax.bar(x + (i - 2) * width, vals, width, label=name, color=colors[i], edgecolor="black", linewidth=0.5)

ax.set_title("Comparaison des Modèles de Recherche d'Information sur NFCorpus (BEIR)", fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(metric_keys, fontsize=11, fontweight="bold")
ax.set_ylabel("Score Moyen", fontsize=11)
ax.set_ylim(0, max(summary_df[metric_keys].values.flatten()) * 1.25)
ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E0", fontsize=9, loc="upper right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3, fontsize=6.5, rotation=45)
plt.tight_layout()
plt.savefig("figures/fig_metrics_comparison.png")
plt.close()

# Plot 2: Latency vs NDCG@10 Pareto Frontier
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
for i, name in enumerate(model_names):
    lat = summary_df.loc[summary_df["Model"] == name, "Latency (ms)"].values[0]
    ndcg = summary_df.loc[summary_df["Model"] == name, "NDCG@10"].values[0]
    ax.scatter(lat, ndcg, s=180, color=colors[i], edgecolor="black", linewidth=1.2, zorder=5)
    offset_x = 0.2 if lat < 5 else -0.8
    offset_y = 0.005
    ax.annotate(f"{name}\n({lat:.2f} ms, {ndcg:.3f})", (lat + offset_x, ndcg + offset_y), fontsize=9, fontweight="bold")

ax.set_title("Compromis Efficacité vs Efficacité (Latency vs NDCG@10)", fontsize=12, fontweight="bold", pad=10)
ax.set_xlabel("Latence moyenne par requête (ms) [Plus bas = Meilleur]", fontsize=10)
ax.set_ylabel("NDCG@10 [Plus haut = Meilleur]", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("figures/fig_latency_vs_ndcg.png")
plt.close()

# Plot 3: Distribution of NDCG@10 across queries
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
box_data = [
    [item["ndcg10_bm25"] for item in detailed_per_query],
    [item["ndcg10_dense"] for item in detailed_per_query],
    [item["ndcg10_hybrid"] for item in detailed_per_query]
]
bp = ax.boxplot(box_data, tick_labels=["BM25", "SBERT Dense", "Hybrid (RRF)"], patch_artist=True, widths=0.45)
box_colors = ["#2B6CB0", "#319795", "#805AD5"]
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("Distribution des scores NDCG@10 par requete sur le jeu de test", fontsize=12, fontweight="bold", pad=10)
ax.set_ylabel("NDCG@10", fontsize=10)
plt.tight_layout()
plt.savefig("figures/fig_query_distributions.png")
plt.close()

print("\n=== 7. ERROR ANALYSIS & QUALITATIVE CASE STUDIES ===")
# Find cases:
# Case A: Dense > BM25 by a large margin (Semantic Match / Vocabulary Mismatch)
# Case B: BM25 > Dense by a large margin (Exact Keyword / Medical Term match)
# Case C: Both Fail (Complex medical question or missing knowledge)

cases_dense_wins = sorted(detailed_per_query, key=lambda x: x["ndcg10_dense"] - x["ndcg10_bm25"], reverse=True)
cases_bm25_wins = sorted(detailed_per_query, key=lambda x: x["ndcg10_bm25"] - x["ndcg10_dense"], reverse=True)
cases_both_fail = [x for x in detailed_per_query if x["ndcg10_bm25"] < 0.1 and x["ndcg10_dense"] < 0.1]

error_cases = {
    "dense_wins": cases_dense_wins[:3],
    "bm25_wins": cases_bm25_wins[:3],
    "both_fail": cases_both_fail[:3]
}

with open("results/error_cases.json", "w", encoding="utf-8") as f:
    json.dump(error_cases, f, indent=2)

print("\nTop Case: Dense Wins (Vocabulary Mismatch resolved by Embeddings):")
c = cases_dense_wins[0]
print(f"Query [{c['qid']}]: '{c['query']}'")
print(f"BM25 NDCG@10: {c['ndcg10_bm25']:.3f} | Dense NDCG@10: {c['ndcg10_dense']:.3f} | Hybrid: {c['ndcg10_hybrid']:.3f}")
print("Gold target doc:", c['gold_cids'][:2])
print("Dense retrieved doc top 1:", c['dense_top3'][0], corpus[c['dense_top3'][0]]['title'])

print("\nTop Case: BM25 Wins (Exact Acronym/Keyword specificity):")
c = cases_bm25_wins[0]
print(f"Query [{c['qid']}]: '{c['query']}'")
print(f"BM25 NDCG@10: {c['ndcg10_bm25']:.3f} | Dense NDCG@10: {c['ndcg10_dense']:.3f} | Hybrid: {c['ndcg10_hybrid']:.3f}")
print("Gold target doc:", c['gold_cids'][:2])
print("BM25 retrieved doc top 1:", c['bm25_top3'][0], corpus[c['bm25_top3'][0]]['title'])

print("\n=== BENCHMARK PIPELINE COMPLETED SUCCESSFULLY! ===")
