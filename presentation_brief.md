# Brief pour Claude Design — présentation de 5 minutes

Colle le texte ci-dessous dans Claude Design, puis ajoute les images du dossier `figures/`.

---

Crée une présentation académique de 8 diapositives (16:9), en français, pour une soutenance orale de 5 minutes en binôme (Étudiant 1 / Étudiant 2, temps de parole équilibré). Style sobre et scientifique : fond clair, une couleur d'accent (vert-bleu), peu de texte, de grands graphiques. Indique sur chaque diapositive qui parle.

**Sujet :** Recherche sémantique par plongements textuels — comparaison de méthodes lexicales (TF-IDF, BM25), de modèles d'embeddings et d'une approche hybride sur le benchmark SciFact (BEIR).

1. **Titre** (Étudiant 1) — « Recherche sémantique par plongements textuels : lexical, dense ou hybride ? » Noms du binôme, cours de NLP, 2026.

2. **Problème** (Étudiant 1) — Le décalage de vocabulaire : une requête et un document pertinent utilisent souvent des mots différents, donc la recherche par mots-clés échoue. Question : les embeddings font-ils mieux, et quand ? Image : `02_3_2_mesure_du_decalage_de_vocabulaire.png` (48 % des paires requête-document partagent moins de la moitié de leurs mots).

3. **Données** (Étudiant 2) — SciFact (BEIR) : 5 183 résumés scientifiques, 809 requêtes d'entraînement, 300 requêtes de test ; une affirmation scientifique → les résumés qui la confirment ou la contredisent. Source : https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip. Image : `01_3_1_exploration_du_corpus.png`.

4. **Méthode** (Étudiant 2) — Baselines : TF-IDF, BM25 (k1, b réglés par validation croisée 5 plis). Modèles denses : all-MiniLM-L6-v2, bge-small, e5-base, bge-base. Hybride : α·dense + (1−α)·BM25, α = 0,9 choisi par validation croisée. Prétraitement lexical : minuscules, mots vides, racinisation Porter ; texte brut pour les embeddings. Le test n'est utilisé qu'une seule fois. Image : `04_7_2_recherche_hybride_bm25_plongements.png`.

5. **Résultats** (Étudiant 1) — nDCG@10 sur le test, IC à 95 % par bootstrap, test de permutation apparié vs BM25 :
   - Hybride 0,764 (+11,6 %, p = 0,0001)
   - bge-base 0,742 (+8,4 %, p = 0,0005)
   - e5-base 0,726 (+6,0 %, p = 0,015)
   - bge-small 0,707 (+3,3 %, non significatif)
   - BM25 0,685 · TF-IDF 0,659 · MiniLM 0,653
   Message : les embeddings ne gagnent pas automatiquement — seuls les modèles plus grands battent clairement BM25. Image : `05_8_resultats_sur_l_ensemble_de_test_300_requetes.png`.

6. **Quand les embeddings aident-ils ?** (Étudiant 1) — Faible chevauchement de mots : BM25 0,33 → bge-base 0,48 (+47 %). Fort chevauchement : BM25 meilleur (0,96 contre 0,91). L'hybride garde le meilleur des deux (0,51 / 0,94). Par requête : embeddings meilleurs 87 fois, égalité 170, BM25 meilleur 43. Images : `08_9_2_effet_du_decalage_de_vocabulaire.png`, `07_9_1_requete_par_requete_qui_gagne.png`.

7. **Limites** (Étudiant 2) — Un seul jeu de données, 300 requêtes, anglais uniquement ; environ un document pertinent par requête (P@10 peu informative) ; certaines annotations semblent bruitées ; les modèles bge/e5 ont peut-être vu des données scientifiques proches à l'entraînement ; coût : bge-base est 5× plus gros que MiniLM et indexe en 56 s contre 44 s. Image : `06_8_1_compromis_qualite_cout.png`.

8. **Conclusion** (les deux) — (1) Les bons modèles d'embeddings battent BM25 de façon significative, surtout quand le vocabulaire diffère. (2) BM25 reste meilleur quand les mots se recoupent. (3) L'hybride est le meilleur compromis. Perspectives : reranking par cross-encoder, fine-tuning sur le domaine. Merci — questions.
