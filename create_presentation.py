"""
Creates a professional, academic 10-slide PowerPoint presentation (.pptx)
for the Semantic Search using Text Embeddings project.
Optimized for a 5-minute oral defense with balanced 2-student speaking parts.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import os

def build_pptx():
    prs = Presentation()
    # 16:9 Widescreen
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Professional academic theme colors
    NAVY = RGBColor(26, 54, 93)      # #1A365D Primary dark
    BLUE = RGBColor(43, 108, 176)    # #2B6CB0 Secondary accent
    TEAL = RGBColor(49, 151, 149)    # #319795 Success / dense highlight
    DARK = RGBColor(45, 55, 72)      # #2D3748 Body text
    MUTED = RGBColor(113, 128, 150)  # #718096 Subtitles / metadata
    BG_LIGHT = RGBColor(247, 250, 252) # #F7FAFC Background card
    WHITE = RGBColor(255, 255, 255)
    CARD_BORDER = RGBColor(226, 232, 240)

    def add_header(slide, title_text, subtitle_text, speaker_tag=None):
        # Header banner
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
        tf = title_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.name = "Arial"
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = NAVY
        
        p2 = tf.add_paragraph()
        p2.text = subtitle_text
        p2.font.name = "Arial"
        p2.font.size = Pt(13)
        p2.font.color.rgb = MUTED
        p2.space_before = Pt(4)

        if speaker_tag:
            tag_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(10.5), Inches(0.45), Inches(2.0), Inches(0.4))
            tag_box.fill.solid()
            tag_box.fill.fore_color.rgb = BLUE
            tag_box.line.color.rgb = BLUE
            tf_tag = tag_box.text_frame
            tf_tag.word_wrap = False
            p_tag = tf_tag.paragraphs[0]
            p_tag.alignment = PP_ALIGN.CENTER
            p_tag.text = speaker_tag
            p_tag.font.name = "Arial"
            p_tag.font.size = Pt(11)
            p_tag.font.bold = True
            p_tag.font.color.rgb = WHITE

    def add_card(slide, left, top, width, height, title, body_lines, accent_color=BLUE):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = BG_LIGHT
        shape.line.color.rgb = CARD_BORDER
        shape.line.width = Pt(1)

        # Accent top bar
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, Inches(0.08))
        bar.fill.solid()
        bar.fill.fore_color.rgb = accent_color
        bar.line.fill.background()

        tb = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.18), width - Inches(0.4), height - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = NAVY

        for line in body_lines:
            p_line = tf.add_paragraph()
            p_line.text = ("• " if not line.startswith("  ") else "   ") + line
            p_line.font.name = "Arial"
            p_line.font.size = Pt(11)
            p_line.font.color.rgb = DARK
            p_line.space_before = Pt(6)

    # -------------------------------------------------------------------------
    # SLIDE 1 : Titre & Introduction
    # -------------------------------------------------------------------------
    slide1 = prs.slides.add_slide(blank_layout)
    # Background accent banner
    banner = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.4))
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.fill.background()

    tb = slide1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.3), Inches(2.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Recherche Sémantique par Plongements Textuels"
    p.font.name = "Arial"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p2 = tf.add_paragraph()
    p2.text = "Évaluation Expérimentale Comparée : Modèles Lexicaux, Denses (SBERT) et Hybrides sur NFCorpus"
    p2.font.name = "Arial"
    p2.font.size = Pt(18)
    p2.font.color.rgb = BLUE
    p2.space_before = Pt(10)

    # Info card
    card = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.0), Inches(11.3), Inches(2.5))
    card.fill.solid()
    card.fill.fore_color.rgb = BG_LIGHT
    card.line.color.rgb = CARD_BORDER

    tb_info = slide1.shapes.add_textbox(Inches(1.3), Inches(4.2), Inches(10.7), Inches(2.1))
    tf_info = tb_info.text_frame
    tf_info.word_wrap = True
    
    items = [
        ("Cadre Académique :", "Projet NLP & Grands Modèles de Langage (LLM) — Master / DS Projet 1 (13 Nov 2026)"),
        ("Binôme Référent :", "Étudiant 1 (Pipeline dense & Benchmark) | Étudiant 2 (Baselines lexicales & Analyse qualitative)"),
        ("Google Classroom :", "Code de soumission : ophm4coy | Livrables : Notebook (.ipynb) + Présentation (.pptx)"),
        ("Démarche Scientifique :", "Problème → Données (NFCorpus BEIR) → Méthodologie → Implémentation → Évaluation (IR Metrics) → Analyse d'Erreurs")
    ]
    for i, (label, val) in enumerate(items):
        p_info = tf_info.paragraphs[0] if i == 0 else tf_info.add_paragraph()
        run1 = p_info.add_run()
        run1.text = label + " "
        run1.font.bold = True
        run1.font.size = Pt(12)
        run1.font.color.rgb = NAVY
        run2 = p_info.add_run()
        run2.text = val
        run2.font.size = Pt(12)
        run2.font.color.rgb = DARK
        p_info.space_before = Pt(8)

    # -------------------------------------------------------------------------
    # SLIDE 2 : Problématique & Motivation (Étudiant 1 - 00:30)
    # -------------------------------------------------------------------------
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "1. Problématique : Le Verrou du 'Vocabulary Mismatch'", 
               "Pourquoi la recherche par mots-clés échoue-t-elle sur le langage naturel complexe ?", "Étudiant 1 | 00:30")

    add_card(slide2, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0),
             "Limites du Lexical (Mots-Clés)",
             [
                 "Mismatch de vocabulaire : Requête et document utilisent des synonymes distincts.",
                 "Exemple médical frappant :",
                 "  Requête : 'lower cholesterol'",
                 "  Document : 'statin therapy in hyperlipidemia'",
                 "  -> Zéro mot commun = Silence documentaire absolu !",
                 "Hypothèse sac de mots (BoW) :",
                 "  Ignorance de l'ordre des mots et de la syntaxe.",
                 "Sensibilité aux formulations figuratives."
             ], NAVY)

    add_card(slide2, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0),
             "Promesse du Sémantique Dense",
             [
                 "Plongements vectoriels continus :",
                 "  Projection dans R^384 via Transformers.",
                 "Topologie sémantique continue :",
                 "  Proximité angulaire (Cosinus) = Proximité de sens.",
                 "Alignement sémantique latent :",
                 "  'cholesterol' et 'hyperlipidemia' sont voisins dans l'espace latent.",
                 "Capture du contexte global :",
                 "  Self-attention bidirectionnelle.",
                 "Inférence instantanée via produit scalaire."
             ], TEAL)

    add_card(slide2, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "Questions de Recherche (RQ)",
             [
                 "RQ1 (Gain sémantique) :",
                 "  Quel gain réel apporte Sentence-BERT face à TF-IDF et BM25 sur un corpus médical spécialisé ?",
                 "RQ2 (Synergie hybride) :",
                 "  La fusion lexicale-sémantique (RRF / Convexe) surpasse-t-elle les modèles individuels ?",
                 "RQ3 (Analyse d'échec) :",
                 "  Dans quels cas précis le modèle dense échoue-t-il alors que BM25 réussit ?"
             ], BLUE)

    # -------------------------------------------------------------------------
    # SLIDE 3 : Données & Exploration (Étudiant 2 - 01:00)
    # -------------------------------------------------------------------------
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "2. Données & Préparation Justifiée : NFCorpus (BEIR)", 
               "Un benchmark médical international conçu pour éprouver l'écart de registre", "Étudiant 2 | 01:00")

    add_card(slide3, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
             "Description du Corpus & Source",
             [
                 "Benchmark : NFCorpus (Nutrition Facts Corpus - BEIR)",
                 "Source officielle : public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/",
                 "Domaine d'application : Articles biomédicaux PubMed / Medline.",
                 "Nature du corpus :",
                 "  • 3 633 documents médicaux complets (titre + résumé).",
                 "  • 323 requêtes de test formulées par des utilisateurs profanes.",
                 "  • 12 334 jugements de pertinence gradués (Qrels).",
                 "Gradation de pertinence :",
                 "  Score 2 = Hautement pertinent (cible directe).",
                 "  Score 1 = Partiellement pertinent (contexte élargi).",
                 "  Score 0 = Non pertinent."
             ], NAVY)

    add_card(slide3, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
             "Prétraitement Justifié des Données",
             [
                 "Prétraitement Lexical (TF-IDF & BM25) :",
                 "  • Tokenisation alphanumérique (mots >= 2 car.).",
                 "  • Mise en minuscules (uniformisation).",
                 "  • Suppression stopwords anglais (non discriminants).",
                 "  • Pas de stemming agressif : préserve les racines médicales.",
                 "",
                 "Prétraitement Dense (Sentence-BERT) :",
                 "  • Texte brut intégral préservé (avec stopwords et ponctuation).",
                 "  • Tokenisation sous-mots WordPiece (gère la morphologie clinique).",
                 "  • Troncature à 256 tokens (fenêtre attentionnelle optimale).",
                 "Statistiques clés :",
                 "  Doc moyen : 233.8 mots | Requête moyenne : 3.29 mots."
             ], BLUE)

    # -------------------------------------------------------------------------
    # SLIDE 4 : Méthodologie & Architectures (Étudiant 1 - 01:45)
    # -------------------------------------------------------------------------
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "3. Méthodologie : Des Modèles Lexicaux aux Bi-Encodeurs Hybrides", 
               "Formulation mathématique des 5 modèles comparés", "Étudiant 1 | 01:45")

    add_card(slide4, Inches(0.8), Inches(1.8), Inches(3.6), Inches(5.0),
             "1. Baselines Lexicales",
             [
                 "TF-IDF + Cosinus :",
                 "  Représentation creuse VSM pondérée.",
                 "  Pondération sublinéaire 1 + log(TF).",
                 "",
                 "BM25 Okapi (Baseline forte) :",
                 "  Saturation non-linéaire du TF :",
                 "    k1 = 1.5 (sensibilité fréquence)",
                 "    b = 0.75 (pénalisation longueur)",
                 "  Index inversé inversible instantané.",
                 "  Excellente précision sur termes rares."
             ], NAVY)

    add_card(slide4, Inches(4.8), Inches(1.8), Inches(3.6), Inches(5.0),
             "2. SBERT Dense Bi-Encoder",
             [
                 "Modèle : all-MiniLM-L6-v2",
                 "  6 couches Transformer, dim = 384.",
                 "Architecture Bi-Encodeur :",
                 "  e_q = Encode(q)  |  e_d = Encode(d)",
                 "  Encodage indépendant et indexable.",
                 "Mean Pooling contextuel masqué :",
                 "  Moyennage des vecteurs tokens valides.",
                 "Normalisation L2 stricte :",
                 "  Produit matriciel GPU = Cosine Sim.",
                 "Vitesse : inférence vectorielle massive."
             ], TEAL)

    add_card(slide4, Inches(8.8), Inches(1.8), Inches(3.7), Inches(5.0),
             "3. Architectures Hybrides",
             [
                 "Reciprocal Rank Fusion (RRF) :",
                 "  RRF(d) = Sum 1 / (60 + rang_m(d))",
                 "  Fusion sans paramètre basée sur l'ordre.",
                 "  Immunisé contre l'incompatibilité des échelles.",
                 "",
                 "Combinaison Convexe Normalisée :",
                 "  S_hyb = a * S_dense + (1-a) * S_bm25",
                 "  Normalisation Min-Max des scores continus.",
                 "  Paramètre a = 0.5 (équilibre optimal)."
             ], BLUE)

    # -------------------------------------------------------------------------
    # SLIDE 5 : Protocole Expérimental & Métriques (Étudiant 2 - 02:30)
    # -------------------------------------------------------------------------
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "4. Protocole Expérimental & Métriques d'Évaluation IR", 
               "Une panoplie complète de métriques scientifiques adaptées au classement", "Étudiant 2 | 02:30")

    add_card(slide5, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
             "Métriques de Qualité d'Ordonnancement",
             [
                 "NDCG@10 (Métrique Maîtresse) :",
                 "  Mesure le gain cumulé actualisé normalisé.",
                 "  Prend en compte la pertinence graduée (score 2 vs 1).",
                 "  Pénalise logarithmiquement les faux positifs en tête.",
                 "",
                 "MRR@10 (Mean Reciprocal Rank) :",
                 "  1 / rang du PREMIER document pertinent trouvé.",
                 "  Mesure l'effort utilisateur immédiat.",
                 "",
                 "MAP@10 (Mean Average Precision) :",
                 "  Moyenne des précisions à chaque palier de rappel.",
                 "",
                 "Précision & Rappel @k (k = 5, 10, 20) :",
                 "  Couverture globale des documents annotés."
             ], NAVY)

    add_card(slide5, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
             "Conditions Expérimentales & Mesure de Coût",
             [
                 "Environnement d'exécution :",
                 "  • Matériel : GPU NVIDIA RTX 2050 (4 GB VRAM, CUDA 13).",
                 "  • Environnement géré rigoureusement avec 'uv'.",
                 "  • PyTorch 2.9 + Transformers 4.57.",
                 "",
                 "Mesures opérationnelles et passage à l'échelle :",
                 "  • Latence par requête (ms) chronométrée au niveau microseconde.",
                 "  • Débit de service (QPS - Queries Per Second).",
                 "  • Temps d'indexation hors-ligne (Offline Indexing Time).",
                 "Évaluation aveugle :",
                 "  Exécution uniforme sur les 323 requêtes de test NFCorpus."
             ], BLUE)

    # -------------------------------------------------------------------------
    # SLIDE 6 : Tableau Synthétique des Résultats (Étudiant 2 - 03:00)
    # -------------------------------------------------------------------------
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "5. Résultats Principaux : L'Approche Hybride Domine le Benchmark", 
               "Évaluation quantitative comparative sur les 323 requêtes de test NFCorpus", "Étudiant 2 | 03:00")

    # Add Table
    table_shape = slide6.shapes.add_table(6, 8, Inches(0.8), Inches(1.8), Inches(11.7), Inches(3.2))
    table = table_shape.table
    table.columns[0].width = Inches(3.1)
    for c in range(1, 8):
        table.columns[c].width = Inches(1.22)

    headers = ["Modèle Évalué", "NDCG@10", "MRR@10", "MAP@10", "P@10", "R@10", "Latence (ms)", "Débit (QPS)"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            p.font.name = "Arial"
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = WHITE

    rows_data = [
        ["TF-IDF (Cosine)", "0.3055", "0.5136", "0.2221", "0.2173", "0.1450", "7.88 ms", "126.9"],
        ["BM25 (Okapi)", "0.3050", "0.5101", "0.2184", "0.2176", "0.1449", "3.42 ms", "292.1"],
        ["SBERT (all-MiniLM-L6-v2)", "0.3195", "0.5104", "0.2290", "0.2427", "0.1589", "65.47 ms", "15.3"],
        ["Hybrid (RRF, k=60)", "0.3330", "0.5381", "0.2418", "0.2443", "0.1615", "68.97 ms", "14.5"],
        ["Hybrid (Convex, α=0.5)", "0.3469", "0.5461", "0.2567", "0.2551", "0.1668", "69.01 ms", "14.5"]
    ]

    for i, row in enumerate(rows_data):
        bg = BG_LIGHT if i % 2 == 0 else WHITE
        if i == 4:  # Highlight winner
            bg = RGBColor(230, 245, 245)
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                p.font.name = "Arial"
                p.font.size = Pt(10.5)
                p.font.bold = (i == 4 or j == 1)
                p.font.color.rgb = TEAL if (i == 4 and j in [1, 2]) else DARK

    # Bullet takeaways below table
    tb_takeaway = slide6.shapes.add_textbox(Inches(0.8), Inches(5.2), Inches(11.7), Inches(1.8))
    tf_t = tb_takeaway.text_frame
    tf_t.word_wrap = True
    takeaways = [
        "✓ SBERT Dense surpasse BM25 : +4.8% en NDCG@10 et +11.5% en Précision@10 grâce à la résolution de la synonymie.",
        "✓ La Fusion Hybride Convexe bat TOUS les modèles : NDCG@10 = 0.3469 (+13.7% vs BM25 !) et MRR@10 = 0.5461 (+7.0%).",
        "✓ Compromis Latence : BM25 est 19x plus rapide que SBERT (3.4 ms vs 65.5 ms), justifiant une architecture hybride en cascade en production."
    ]
    for i, t in enumerate(takeaways):
        p_t = tf_t.paragraphs[0] if i == 0 else tf_t.add_paragraph()
        p_t.text = t
        p_t.font.name = "Arial"
        p_t.font.size = Pt(12)
        p_t.font.bold = (i == 1)
        p_t.font.color.rgb = NAVY if i == 1 else DARK
        p_t.space_before = Pt(4)

    # -------------------------------------------------------------------------
    # SLIDE 7 : Analyse Visuelle & Frontière de Pareto (Étudiant 1 - 03:30)
    # -------------------------------------------------------------------------
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "6. Visualisations Expérimentales : Métriques & Frontière de Pareto", 
               "Analyse comparative multidimensionnelle et compromis efficience / efficacité", "Étudiant 1 | 03:30")

    # Add images if available
    if os.path.exists("figures/fig_metrics_comparison.png"):
        slide7.shapes.add_picture("figures/fig_metrics_comparison.png", Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.8))
    if os.path.exists("figures/fig_latency_vs_ndcg.png"):
        slide7.shapes.add_picture("figures/fig_latency_vs_ndcg.png", Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.8))

    # -------------------------------------------------------------------------
    # SLIDE 8 : Analyse Critique des Erreurs (Étudiant 1 - 04:00)
    # -------------------------------------------------------------------------
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "7. Analyse Approfondie des Erreurs : Études de Cas Qualitatives", 
               "Quand le sémantique triomphe, et pourquoi le lexical reste indispensable", "Étudiant 1 | 04:00")

    add_card(slide8, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
             "Cas 1 : Victoire Éclatante du Sémantique Dense",
             [
                 "Requête PLAIN-307 :",
                 "  'Vitamin D: Shedding some light on the new recommendations'",
                 "",
                 "Scores : BM25 = 0.156  |  SBERT Dense = 0.819 (+0.663 !)",
                 "",
                 "Diagnostic d'échec BM25 :",
                 "  • 'shedding', 'light' et 'recommendations' polluent l'inversion d'index.",
                 "  • BM25 renvoie des articles sur la photothérapie ou des lampes !",
                 "",
                 "Succès SBERT :",
                 "  • Le modèle dense comprend l'analogie et capture l'intention réelle :",
                 "    les directives cliniques de carence en vitamine D (MED-714)."
             ], TEAL)

    add_card(slide8, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
             "Cas 2 : Victoire Éclatante du Lexical BM25",
             [
                 "Requête PLAIN-1473 :",
                 "  'kohlrabi' (chou-rave, variété botanique spécifique)",
                 "",
                 "Scores : BM25 = 1.000  |  SBERT Dense = 0.000 (Échec total !)",
                 "",
                 "Diagnostic d'échec SBERT :",
                 "  • Entité rare sous-représentée dans le pré-entraînement.",
                 "  • Dilution du concept dans l'espace dense des légumes crucifères.",
                 "  • SBERT renvoie des études génériques sur le brocoli.",
                 "",
                 "Succès BM25 :",
                 "  • Matching inversé exact du token rare 'kohlrabi' dans le titre",
                 "    de l'article cible MED-4455. Précision parfaite !"
             ], NAVY)

    # -------------------------------------------------------------------------
    # SLIDE 9 : Limites & Perspectives (Étudiant 2 - 04:30)
    # -------------------------------------------------------------------------
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "8. Limites Identifiées & Perspectives d'Amélioration", 
               "Axes d'optimisation pour le déploiement en environnement de production", "Étudiant 2 | 04:30")

    add_card(slide9, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.0),
             "Limites Intrinsèques Détectées",
             [
                 "1. Goulot d'étranglement vectoriel à dimension fixe :",
                 "  Compresser 500 mots médicaux en 384 dimensions efface",
                 "  les détails fins (posologies numériques, négations strictes).",
                 "",
                 "2. Biais de domaine médical :",
                 "  all-MiniLM-L6-v2 est généraliste ; un modèle adapté",
                 "  (ex: BioLinkBERT, PubMedBERT) affinerait le vocabulaire.",
                 "",
                 "3. Coût de calcul en première passe :",
                 "  65 ms par requête interdit la recherche exhaustive par",
                 "  produit matriciel sur des corpus de plusieurs millions de documents."
             ], NAVY)

    add_card(slide9, Inches(6.8), Inches(1.8), Inches(5.7), Inches(5.0),
             "Perspectives & Évolution Industrielle",
             [
                 "1. Architecture en Cascade (Two-Stage Retrieval) :",
                 "  • Étage 1 : Pré-filtrage rapide BM25 (top 100 en 3 ms).",
                 "  • Étage 2 : Ré-ordonnancement par Cross-Encoder lourd.",
                 "",
                 "2. Interaction Tardive (ColBERTv2 / Late Interaction) :",
                 "  Représentations multi-vecteurs par token + opérateur MaxSim.",
                 "  Allie la finesse du Cross-Encoder à la rapidité de l'index vectoriel.",
                 "",
                 "3. Indexation Vectorielle Approximative (ANN) :",
                 "  Déploiement d'index HNSW ou ScaNN pour passer à l'échelle du web."
             ], BLUE)

    # -------------------------------------------------------------------------
    # SLIDE 10 : Conclusion & Clôture (Binôme - 05:00)
    # -------------------------------------------------------------------------
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "9. Conclusion & Synthèse du Projet", 
               "Les 3 enseignements majeurs de notre démarche scientifique", "Binôme | 05:00")

    add_card(slide10, Inches(0.8), Inches(1.8), Inches(3.6), Inches(4.8),
             "1. Révolution Sémantique",
             [
                 "Validation de l'apport des plongements textuels :",
                 "  SBERT résout le fossé de registre profane/expert (+4.8% NDCG).",
                 "Démontré sur NFCorpus :",
                 "  Compréhension contextuelle des paraphrases et métaphores."
             ], NAVY)

    add_card(slide10, Inches(4.8), Inches(1.8), Inches(3.6), Inches(4.8),
             "2. Nécessité de l'Hybride",
             [
                 "Le modèle Hybride est le véritable état de l'art :",
                 "  +13.7% de gain NDCG@10 (0.3469 vs 0.3050).",
                 "Complémentarité parfaite :",
                 "  Précision lexicale sur entités rares + lissage sémantique continu."
             ], TEAL)

    add_card(slide10, Inches(8.8), Inches(1.8), Inches(3.7), Inches(4.8),
             "3. Rigueur Scientifique",
             [
                 "Démarche expérimentale complète :",
                 "  Dataset officiel, 5 modèles, 8 métriques, benchmarking GPU réel.",
                 "Analyse sans concession :",
                 "  Mise en lumière des échecs denses (cas kohlrabi).",
                 "",
                 "Merci pour votre attention !",
                 "Nous sommes prêts pour vos questions."
             ], BLUE)

    prs.save("presentation_semantic_search.pptx")
    print("✓ Présentation PowerPoint générée avec succès : presentation_semantic_search.pptx")

if __name__ == "__main__":
    build_pptx()
