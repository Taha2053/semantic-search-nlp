# 🎙️ Script de Présentation Orale (5 Minutes Chrono - Soutenance en Binôme)
## Sujet : *Recherche Sémantique par Plongements Textuels (Semantic Search using Dense Embeddings)*
### Cadre : Projet NLP / Master — Date : 13 Novembre 2026 — Google Classroom : `ophm4coy`

---

### ⏱️ Répartition Globale du Temps de Parole (Équilibre Strict 50% / 50%)
| Intervenant | Rôles & Sections Présentées | Durée Cumulée |
| :--- | :--- | :--- |
| **Étudiant 1** | Introduction, Problématique, Modèles (SBERT/Hybride), Visualisations, Étude de cas (Erreurs) | **2 min 30 s (150 s)** |
| **Étudiant 2** | Données (NFCorpus), Prétraitement, Protocole IR, Résultats chiffrés, Limites, Perspectives | **2 min 30 s (150 s)** |
| **Total** | **Soutenance complète en 10 Diapositives** | **5 min 00 s (300 s)** |

---

## 🕒 Chronologie Détaillée Slide par Slide

### 🎞️ Diapositive 1 : Titre & Introduction Générale
**Temps :** 00:00 – 00:30 (Durée : 30 s)  
**Orateur :** **Étudiant 1**  
**Visuel :** Titre, noms du binôme, cadre académique et code Google Classroom (`ophm4coy`).

> **Étudiant 1 :**  
> *"Bonjour Madame, bonjour à tous.  
> Avec mon binôme [Nom de l'Étudiant 2], nous avons le plaisir de vous présenter notre projet de NLP intitulé : **Recherche Sémantique par Plongements Textuels**.  
> Notre démarche a été guidée par une exigence scientifique stricte : nous n'avons pas simplement appliqué une bibliothèque existante, mais nous avons conçu une démarche expérimentale rigoureuse de bout en bout : formulation du problème, analyse d'un benchmark académique international, implémentation de baselines lexicales et de modèles denses sur GPU, évaluation quantitative multi-métriques, et enfin analyse qualitative des modes d'échec."*

---

### 🎞️ Diapositives 2 & 3 : Problématique & Données (NFCorpus BEIR)
**Temps :** 00:30 – 01:15 (Durée : 45 s)  
**Orateur :** **Étudiant 2**  
**Visuel :** Cartes des limites lexicales vs plongements denses, statistiques descriptives du corpus NFCorpus.

> **Étudiant 2 :**  
> *"Merci [Nom Étudiant 1].  
> Le verrou fondamental des moteurs de recherche classiques comme le TF-IDF ou BM25 est le **Vocabulary Mismatch** : si un utilisateur recherche 'how to lower high cholesterol' et qu'un article médical traite de 'statin therapy in hyperlipidemia', les modèles par mots-clés constatent zéro terme en commun et renvoient un silence documentaire total.  
> Pour tester ce problème, nous avons choisi le benchmark officiel **NFCorpus**, issu du standard international **BEIR**.  
> Ce dataset réunit 3 633 documents médicaux de PubMed et 323 requêtes de test rédigées en langage profane par de vrais patients, associées à plus de 12 000 jugements de pertinence gradués.  
> Côté prétraitement : pour le lexical, nous avons filtré les mots vides sans altérer les racines médicales ; pour le modèle Transformer dense, nous avons préservé le texte brut intégral afin de conserver toute la structure syntaxique pour les mécanismes d'attention."*

---

### 🎞️ Diapositive 4 : Méthodologie & Architectures
**Temps :** 01:15 – 02:00 (Durée : 45 s)  
**Orateur :** **Étudiant 1**  
**Visuel :** Comparaison des 5 architectures : TF-IDF, BM25, SBERT Bi-Encoder, Hybride RRF et Hybride Convexe.

> **Étudiant 1 :**  
> *"Pour répondre à cette problématique, nous avons implémenté et comparé 5 systèmes complémentaires :  
> 1. D'abord deux baselines lexicales : le modèle vectoriel **TF-IDF**, et **BM25 Okapi** qui module la saturation des termes fréquents et normalise la longueur des documents.  
> 2. Ensuite, notre modèle dense avancé : un **Bi-Encoder Sentence-BERT** reposant sur `all-MiniLM-L6-v2`. Le modèle projette chaque document et chaque requête dans un espace continu de dimension 384 via un Mean Pooling normalisé L2. La recherche en ligne se réduit alors à un produit scalaire ultra-rapide sur GPU.  
> 3. Enfin, nous avons conçu deux approches hybrides : la **Fusion par Rangs Réciproques (RRF)**, qui agrège les classements ordonnés sans paramètre, et une **Combinaison Convexe** qui fusionne linéairement les scores normalisés."*

---

### 🎞️ Diapositives 5 & 6 : Protocole Expérimental & Résultats Principaux
**Temps :** 02:00 – 02:45 (Durée : 45 s)  
**Orateur :** **Étudiant 2**  
**Visuel :** Grand tableau comparatif des 5 modèles avec NDCG@10, MRR@10, Précision, Rappel, Latence et QPS.

> **Étudiant 2 :**  
> *"Venons-en au protocole et aux résultats.  
> Nous avons mesuré les métriques canoniques de la recherche d'information : le **NDCG@10**, qui prend en compte la pertinence graduée, le **MRR@10**, le **MAP**, la Précision et le Rappel, ainsi que la latence d'inférence.  
> Voici nos trois conclusions majeures :  
> Première observation : SBERT dense surpasse nettement BM25 avec un NDCG@10 de **0.3195** contre **0.3050**, et un gain spectaculaire de **+11.5% en Précision@10**.  
> Deuxième observation, cruciale : **notre modèle Hybride Convexe bat tous les systèmes** en établissant l'état de l'art à **0.3469 en NDCG@10**, soit un bond de **+13.7%** par rapport à BM25.  
> Troisième observation : sur le plan opérationnel, BM25 s'exécute en **3.4 ms** par requête contre **65.5 ms** pour SBERT. L'hybridation apporte donc un gain qualitatif majeur, mais doit être articulée avec discernement."*

---

### 🎞️ Diapositives 7 & 8 : Visualisations & Analyse Critique des Erreurs
**Temps :** 02:45 – 03:45 (Durée : 60 s)  
**Orateur :** **Étudiant 1**  
**Visuel :** Barplot multi-métriques, Frontière de Pareto Latence vs NDCG, et les deux cas d'étude qualitatifs (`Vitamin D` vs `kohlrabi`).

> **Étudiant 1 :**  
> *"La démarche scientifique exige de regarder sous le capot des chiffres globaux.  
> Comme vous le voyez sur notre frontière de Pareto, l'hybride se situe au sommet de l'efficacité, tandis que BM25 offre le débit maximal.  
> Mais l'analyse la plus passionnante concerne nos **études de cas qualitatifs** :  
> Regardez le Cas 1 : sur la requête *'Vitamin D: Shedding some light on the new recommendations'*, BM25 s'effondre avec un NDCG de **0.156** car il se focalise sur les mots métaphoriques 'shedding' et 'light', renvoyant des articles sur les lampes ! SBERT, lui, comprend la métaphore et atteint **0.819** en retrouvant directement les recommandations médicales.  
> Mais à l'inverse, regardez le Cas 2 : sur la requête *'kohlrabi'* (le chou-rave), SBERT obtient **0.000** ! L'entité rare a été diluée dans l'espace des légumes verts, alors que BM25 obtient un score parfait de **1.000** grâce au matching exact du terme dans l'index inversé.  
> C'est la preuve expérimentale irréfutable que la sémantique et le lexical sont deux signaux indissociables."*

---

### 🎞️ Diapositive 9 : Limites & Perspectives
**Temps :** 03:45 – 04:30 (Durée : 45 s)  
**Orateur :** **Étudiant 2**  
**Visuel :** Carte des limites détectées (goulot d'étranglement vectoriel) et perspectives (ColBERTv2, Two-Stage Reranking, FAISS).

> **Étudiant 2 :**  
> *"Ces constats nous amènent aux limites et perspectives de notre travail.  
> La limite principale du bi-encodeur réside dans le goulot d'étranglement vectoriel : compresser 300 mots dans un unique vecteur de 384 dimensions efface inévitablement les informations numériques fines, comme une posologie en milligrammes ou les négations strictes.  
> Pour un passage à l'échelle industrielle, trois perspectives s'imposent :  
> 1. Déployer une **architecture en cascade (Retrieve & Re-rank)** : filtrer les 100 meilleurs candidats avec BM25 en 3 ms, puis réordonner avec un Cross-Encoder lourd.  
> 2. Adopter le paradigme d'**Interaction Tardive de ColBERTv2**, qui conserve les représentations de chaque token tout en conservant la vitesse du bi-encodeur via l'opérateur MaxSim.  
> 3. Utiliser des index vectoriels approximatifs de type HNSW ou FAISS pour traiter des millions de documents sans latence excessive."*

---

### 🎞️ Diapositive 10 : Conclusion & Clôture
**Temps :** 04:30 – 05:00 (Durée : 30 s)  
**Orateurs :** **Étudiant 1 & Étudiant 2 (Passage de relais)**  
**Visuel :** Synthèse en 3 points cardinaux et message de remerciement.

> **Étudiant 1 :**  
> *"En conclusion, ce projet a démontré que si les plongements textuels denses transforment la recherche d'information en résolvant le mismatch lexical..."*  
>   
> **Étudiant 2 :**  
> *"...c'est véritablement l'alliance hybride entre la précision chirurgicale du lexical et la puissance associative du sémantique qui définit l'état de l'art.  
> Tous nos livrables — le notebook interactif avec son code GPU, les graphiques et cette présentation — sont prêts et déposés.  
> Nous vous remercions pour votre attention et nous serons ravis de répondre à vos questions !"*

---

## 🎯 Guide de Préparation aux Questions du Jury (FAQ Probable)

### Q1. *"Pourquoi avoir choisi NFCorpus plutôt qu'un dataset plus générique comme MS MARCO ou Quora ?"*
> **Réponse (Étudiant 2) :**  
> *"NFCorpus présente une caractéristique académique idéale : un fossé de vocabulaire asymétrique entre des questions d'utilisateurs profanes et des articles médicaux spécialisés. C'est le terrain d'épreuve le plus difficile et le plus démonstratif pour comparer le matching exact et la recherche sémantique dense."*

### Q2. *"Pourquoi all-MiniLM-L6-v2 et non pas un modèle plus grand comme BERT-Large ou un LLM 7B ?"*
> **Réponse (Étudiant 1) :**  
> *"all-MiniLM-L6-v2 offre un ratio efficience/performance optimal : 22 millions de paramètres et une dimension 384, ce qui permet une inférence ultra-rapide et légère sur notre GPU de 4 Go de VRAM, tout en obtenant des scores compétitifs sur le benchmark BEIR face à des modèles 10 fois plus lourds."*

### Q3. *"Comment avez-vous réglé le paramètre alpha dans la combinaison hybride convexe ?"*
> **Réponse (Étudiant 2) :**  
> *"Nous avons normalisé les scores Min-Max pour ramener les scores BM25 et cosinus dans le même intervalle [0, 1]. Nous avons testé différentes valeurs de alpha et la valeur médiane alpha = 0.5 offre un équilibre parfait entre le rappel sémantique et la précision lexicale."*

### Q4. *"Quelle est la différence fondamentale entre un Bi-Encoder et un Cross-Encoder ?"*
> **Réponse (Étudiant 1) :**  
> *"Le Bi-Encoder encode la requête et le document séparément, ce qui permet d'indexer les documents hors-ligne une fois pour toutes et de chercher en quelques millisecondes. Le Cross-Encoder passe la paire complète dans le Transformer avec de la cross-attention entre tous les mots ; il est beaucoup plus précis, mais sa complexité en O(N) le rend impraticable pour chercher dans des milliers de documents en temps réel."*
