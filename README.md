# 🎓 Semantic Search using Text Embeddings
## *Projet Appliqué NLP & Grands Modèles de Langage (LLM)*

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/Manager-uv-blueviolet.svg)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%20CUDA-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![Benchmark: BEIR](https://img.shields.io/badge/Benchmark-BEIR%20NFCorpus-green.svg)](https://github.com/beir-cellar/beir)

---

### 📌 Cadre & Modalités Académiques
- **Matière :** Traitement Automatique du Langage Naturel (NLP) & LLM
- **Sujet choisi :** *Semantic Search using Text Embeddings*
- **Échéance :** 13 Novembre 2026 (Projet 1 - Note de DS)
- **Code Google Classroom :** `ophm4coy`
- **Démarche scientifique appliquée :** *Problème $\rightarrow$ Données $\rightarrow$ Méthodologie $\rightarrow$ Implémentation $\rightarrow$ Évaluation $\rightarrow$ Analyse critique & Erreurs*.

---

## 📦 Livrables Complets du Projet

Conformément aux exigences strictes du sujet, tous les livrables obligatoires sont disponibles à la racine :

| Livrable | Fichier | Description |
| :--- | :--- | :--- |
| **1) Livrable Principal (Notebook)** | [`Semantic_Search_NLP_Project.ipynb`](Semantic_Search_NLP_Project.ipynb) | Notebook Jupyter exhaustif, structuré, auto-porteur et **entièrement exécuté** avec visualisations haute résolution et tableaux intégrés. |
| **2) Support de Présentation (PPTX)** | [`presentation_semantic_search.pptx`](presentation_semantic_search.pptx) | Diaporama PowerPoint 16:9 professionnel (10 slides) prêt pour la soutenance. |
| **3) Script Oral Minuté (5 minutes)** | [`presentation_script_5min.md`](presentation_script_5min.md) | Guide mot à mot minuté (00:00 - 05:00) avec répartition équilibrée à 50% / 50% pour un binôme (Étudiant 1 & Étudiant 2). |
| **4) Pipeline Expérimental Reproductible** | [`benchmark_pipeline.py`](benchmark_pipeline.py) | Script autonome d'évaluation complète sur GPU avec génération des figures et extraction des cas d'erreurs. |
| **5) Source & Données du Dataset** | [`data/nfcorpus/`](data/nfcorpus/) | Benchmark médical international NFCorpus téléchargé et vérifié. |

---

## 🎯 Synthèse Scientifique des Résultats

L'évaluation porte sur les **323 requêtes de test annotées** du benchmark officiel **NFCorpus (BEIR)** face à un corpus de **3 633 documents médicaux** (12 334 jugements de pertinence Qrels gradués).

### 📊 Tableau Comparatif des Performances IR

| Modèle Évalué | NDCG@10 | MRR@10 | MAP@10 | P@10 | R@10 | Latence par requête | Débit (QPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF (Cosine)** | 0.3055 | 0.5136 | 0.2221 | 0.2173 | 0.1450 | 7.88 ms | 126.9 |
| **BM25 (Okapi)** | 0.3050 | 0.5101 | 0.2184 | 0.2176 | 0.1449 | **3.42 ms** | **292.1** |
| **SBERT (`all-MiniLM-L6-v2`)** | 0.3195 | 0.5104 | 0.2290 | 0.2427 | 0.1589 | 65.47 ms | 15.3 |
| **Hybrid (RRF, $k=60$)** | 0.3330 | 0.5381 | 0.2418 | 0.2443 | 0.1615 | 68.97 ms | 14.5 |
| **Hybrid (Convexe, $\alpha=0.5$)** | **0.3469** | **0.5461** | **0.2567** | **0.2551** | **0.1668** | 69.01 ms | 14.5 |

### 🔬 Enseignements Scientifiques Clés
1. **Gain Sémantique du Bi-Encodeur :** Le modèle dense SBERT surpasse BM25 (+4.8% en NDCG@10 et +11.5% en Précision@10) en résolvant le **Vocabulary Mismatch** (synonymie, variations morphologiques et langage profane vs clinique).
2. **Domination de l'Hybridation :** Le modèle hybride convexe établit le score de référence (**NDCG@10 = 0.3469, soit +13.7% par rapport à BM25**), prouvant que les représentations creuses et denses apportent des signaux hautement complémentaires.
3. **Analyse Critique des Erreurs :**
   - **Sémantique triomphant :** Requête *"Vitamin D: Shedding some light on the new recommendations"* (BM25 = 0.156 vs SBERT = **0.819**). SBERT comprend la métaphore sans se faire piéger par les mots de surface.
   - **Lexical triomphant :** Requête *"kohlrabi"* (entité rare) (SBERT = 0.000 vs BM25 = **1.000**). Les modèles denses souffrent d'une dilution des entités sous-représentées, où l'inversion d'index exacte reste imbattable.

---

## 🚀 Reproduction Rapide avec `uv`

Ce projet utilise **`uv`** comme gestionnaire d'environnement et de dépendances.

### 1. Cloner ou ouvrir le répertoire
```bash
cd project-nlp
```

### 2. Exécuter le pipeline de benchmark
```bash
uv run --no-sync python3 benchmark_pipeline.py
```

### 3. Lancer le Notebook Jupyter
```bash
uv run --no-sync jupyter lab Semantic_Search_NLP_Project.ipynb
# ou ouvrir directement dans Google Colab ou VS Code
```

### 4. Régénérer la présentation PowerPoint si besoin
```bash
uv run --no-sync python3 create_presentation.py
```

---

## 🔗 Source et Citation du Dataset

- **Dataset :** NFCorpus (Nutrition Facts Corpus)
- **Collection :** BEIR Benchmark (Benchmarking Information Retrieval)
- **URL officielle :** [https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip)
- **Référence :**
  > Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models.* NeurIPS 2021 Datasets & Benchmarks Track.

---

## 🛡️ Déclaration d'Utilisation des Outils IA
*Conformément aux consignes de transparence de l'enseignante, des outils d'IA générative ont été consultés pour l'assistance à la structuration documentaire et le prototypage du code. L'ensemble de la conception scientifique, des choix algorithmiques, des implémentations de modèles, des vérifications expérimentales sur GPU et des analyses critiques ont été intégralement menés et validés par les étudiants.*
