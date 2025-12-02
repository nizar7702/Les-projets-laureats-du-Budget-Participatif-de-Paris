import sqlite3
import pandas as pd
import numpy as np
import requests
import re
import json
import threading
import unicodedata
import random
from typing import Tuple, Dict, Any, Optional, Generator
import difflib

ARRONDISSEMENT_MAP = {
    # 1er
    "1": 1, "1er": 1, "1e": 1, "1eme": 1, "1ème": 1, "1ieme": 1, "1iéme": 1,
    "premier": 1, "paris 1er": 1, "paris 1e": 1,

    # 2e
    "2": 2, "2e": 2, "2eme": 2, "2ème": 2, "2ieme": 2, "2iéme": 2,
    "deuxieme": 2, "deuxième": 2, "paris 2e": 2,

    # 3e
    "3": 3, "3e": 3, "3eme": 3, "3ème": 3, "3ieme": 3, "3iéme": 3,
    "troisieme": 3, "troisième": 3, "paris 3e": 3,

    # 4e
    "4": 4, "4e": 4, "4eme": 4, "4ème": 4, "4ieme": 4, "4iéme": 4,
    "quatrieme": 4, "quatrième": 4, "paris 4e": 4,

    # 5e
    "5": 5, "5e": 5, "5eme": 5, "5ème": 5, "5ieme": 5, "5iéme": 5,
    "cinquieme": 5, "cinquième": 5, "paris 5e": 5,

    # 6e
    "6": 6, "6e": 6, "6eme": 6, "6ème": 6, "6ieme": 6, "6iéme": 6,
    "sixieme": 6, "sixième": 6, "paris 6e": 6,

    # 7e
    "7": 7, "7e": 7, "7eme": 7, "7ème": 7, "7ieme": 7, "7iéme": 7,
    "septieme": 7, "septième": 7, "paris 7e": 7,

    # 8e
    "8": 8, "8e": 8, "8eme": 8, "8ème": 8, "8ieme": 8, "8iéme": 8,
    "huitieme": 8, "huitième": 8, "paris 8e": 8,

    # 9e
    "9": 9, "9e": 9, "9eme": 9, "9ème": 9, "9ieme": 9, "9iéme": 9,
    "neuvieme": 9, "neuvième": 9, "paris 9e": 9,

    # 10e
    "10": 10, "10e": 10, "10eme": 10, "10ème": 10, "10ieme": 10, "10iéme": 10,
    "dixieme": 10, "dixième": 10, "paris 10e": 10,

    # 11e
    "11": 11, "11e": 11, "11eme": 11, "11ème": 11, "11ieme": 11, "11iéme": 11,
    "onzieme": 11, "onzième": 11, "paris 11e": 11,

    # 12e
    "12": 12, "12e": 12, "12eme": 12, "12ème": 12, "12ieme": 12, "12iéme": 12,
    "douzieme": 12, "douzième": 12, "paris 12e": 12,

    # 13e
    "13": 13, "13e": 13, "13eme": 13, "13ème": 13, "13ieme": 13, "13iéme": 13,
    "treizieme": 13, "treizième": 13, "paris 13e": 13,

    # 14e
    "14": 14, "14e": 14, "14eme": 14, "14ème": 14, "14ieme": 14, "14iéme": 14,
    "quatorzieme": 14, "quatorzième": 14, "paris 14e": 14,

    # 15e
    "15": 15, "15e": 15, "15eme": 15, "15ème": 15, "15ieme": 15, "15iéme": 15,
    "quinzieme": 15, "quinzième": 15, "paris 15e": 15,

    # 16e
    "16": 16, "16e": 16, "16eme": 16, "16ème": 16, "16ieme": 16, "16iéme": 16,
    "seizieme": 16, "seizième": 16, "paris 16e": 16,

    # 17e
    "17": 17, "17e": 17, "17eme": 17, "17ème": 17, "17ieme": 17, "17iéme": 17,
    "dix-septieme": 17, "dix-septième": 17, "paris 17e": 17,

    # 18e
    "18": 18, "18e": 18, "18eme": 18, "18ème": 18, "18ieme": 18, "18iéme": 18,
    "dix-huitieme": 18, "dix-huitième": 18, "paris 18e": 18,

    # 19e
    "19": 19, "19e": 19, "19eme": 19, "19ème": 19, "19ieme": 19, "19iéme": 19,
    "dix-neuvieme": 19, "dix-neuvième": 19, "paris 19e": 19,

    # 20e
    "20": 20, "20e": 20, "20eme": 20, "20ème": 20, "20ieme": 20, "20iéme": 20,
    "vingtieme": 20, "vingtième": 20, "paris 20e": 20
}
STATUS_MAP = {
    # FIN
    "fin": "FIN", "fini": "FIN", "finis": "FIN", "termine": "FIN", "terminé": "FIN",
    "terminée": "FIN", "acheve": "FIN", "achevé": "FIN", "achevée": "FIN",
    "clos": "FIN", "fermé": "FIN", "finalise": "FIN", "finalisé": "FIN", "finalisée": "FIN",

    # LIVRAISON
    "livraison": "LIVRAISON", "livre": "LIVRAISON", "livré": "LIVRAISON", "livrée": "LIVRAISON",
    "remis": "LIVRAISON", "remise": "LIVRAISON",

    # ETUDES
    "etude": "ETUDES", "etudes": "ETUDES", "etudier": "ETUDES", "analyse": "ETUDES",
    "conception": "ETUDES", "planification": "ETUDES",

    # TRAVAUX
    "travaux": "TRAVAUX", "construction": "TRAVAUX", "chantier": "TRAVAUX",
    "batir": "TRAVAUX", "batiment": "TRAVAUX", "réalisation": "TRAVAUX",

    # NON DEMARRE
    "non demarre": "NON DEMARRE", "pas commence": "NON DEMARRE", "pas commencé": "NON DEMARRE",
    "non commencé": "NON DEMARRE", "non debuté": "NON DEMARRE", "non débuté": "NON DEMARRE",

    # PROCEDURES
    "procedure": "PROCEDURES", "procedures": "PROCEDURES", "administratif": "PROCEDURES",
    "juridique": "PROCEDURES", "autorisation": "PROCEDURES", "validation": "PROCEDURES",

    # ABANDONNÉ
    "abandonne": "ABANDONNÉ", "abandonné": "ABANDONNÉ", "abandonnée": "ABANDONNÉ",
    "annule": "ABANDONNÉ", "annulé": "ABANDONNÉ", "annulée": "ABANDONNÉ",
    "suspendu": "ABANDONNÉ", "suspendue": "ABANDONNÉ"
}
SEEDS = {
    "Prévention et sécurité": [
        "sécurité","sécurisation","danger","risque","éclairage","lampadaire","luminaire",
        "caméra","vidéosurveillance","police","incendie","pompier","signalisation",
        "passage piéton","radar","alarme","sirène","feu tricolore","ralentisseur","dos d’âne"
    ],
    "Cadre de vie": [
        "cadre de vie","qualité de vie","espaces publics","urbanisme","aménagement","place",
        "square","mobilier urbain","banc","aire de repos","aire de jeux","parvis","esplanade",
        "rénovation","embellissement","poteau","fontaine","kiosque","terrasse","pergola"
    ],
    "Environnement": [
        "environnement","nature","biodiversité","jardin","parc public","écologie","arbre","arbres",
        "plantation","verdissement","compost","eau","toiture végétale","mur végétal","végétalisation",
        "climat","développement durable","recyclage","pollution","air","énergie renouvelable",
        "solaire","photovoltaïque","prairie fleurie","nichoir","hôtel à insectes"
    ],
    "Sport": [
        "sport","football","foot","gymnase","terrain","stade","basket","handball","tennis",
        "natation","piscine","running","course","dojo","arts martiaux","rugby","volley",
        "badminton","patinoire","skatepark","escalade","fitness","musculation","city stade"
    ],
    "Culture et patrimoine": [
        "culture","patrimoine","bibliothèque","médiathèque","musée","spectacle","archive","exposition",
        "art","cinéma","festival","théâtre","concert","musique","danse","peinture","sculpture",
        "parc d’attraction","parc de loisirs","divertissement","galerie","salle de spectacle",
        "conservatoire"
    ],
    "Solidarités": [
        "solidarité","inclusion","entraide","accessibilité","handicap","aide alimentaire",
        "épicerie solidaire","réfugiés","personnes âgées","soutien","logement social","sans-abri",
        "précarité","accueil de jour"
    ],
    "Education et jeunesse": [
        "éducation","école","crèche","jeunesse","collège","lycée","université","activité éducative",
        "périscolaire","bibliobus","formation","alphabétisation","enseignement","cantine",
        "soutien scolaire"
    ],
    "Mobilités": [
        "mobilité","transport","vélo","piste cyclable","trottoir","stationnement","parking","bus",
        "tram","métro","voie verte","circulation","piéton","piétonisation","route","chaussée",
        "signalisation routière","borne de recharge","recharge électrique"
    ],
    "Propreté": [
        "propreté","déchet","déchets","nettoyage","sensibilisation","tri","poubelles","corbeilles",
        "dépôts sauvages","ramassage","balayage","ordures","collecte","benne","lavage voirie"
    ],
    "Santé": [
        "santé","soin","hôpital","prévention sanitaire","centre de santé","clinique","pharmacie",
        "vaccination","médecin","consultation","urgences","santé publique","infirmerie","cabinet médical"
    ],
    "Attractivité et emploi": [
        "emploi","travail","commerce","attractivité","économie","artisanat","marché","tourisme",
        "entrepreneuriat","incubateur","startup","innovation","industrie","foire","boutique",
        "attraction touristique","pôle économique"
    ]
}

# utils.py
# Complete utility module WITHOUT any map declarations (ARRONDISSEMENT_MAP, STATUS_MAP, SEEDS, THEMATIQUE_MAP).
# These maps are assumed to be defined elsewhere in your project.


import hashlib
import time

# -----------------------
# Configuration LLM / DB
# -----------------------
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "phi3:mini"

# Cache en mémoire pour suggestions
_SUGGESTIONS_CACHE: Dict[str, str] = {}

# -----------------------
# SQL helpers
# -----------------------
def load_dataframe_from_sql(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Charge un DataFrame depuis une base SQLite. Conserve les colonnes utiles si présentes.
    """
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()

    features = [
        "titre_projet_gagnant",
        "thematique",
        "arrondissement_projet_gagnant",
        "budget_global_projet_gagnant",
        "avancement_projet",
        "edition",
    ]
    existing = [c for c in features if c in df.columns]
    return df[existing] if existing else df

# -----------------------
# Helpers de normalisation
# -----------------------
def normalize_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[’'`]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s
def variants(term: str) -> set:
    """
    Generate normalized variants of a seed term (dash/apostrophe/space forms, plural heuristic).
    Relies on normalize_text already defined in utils.py.
    """
    t = normalize_text(term)
    forms = {t, t.replace("-", " "), t.replace("’", " "), t.replace("'", " ")}
    if " " not in t and not t.endswith("s"):
        forms.add(t + "s")
    return forms

def build_thematique_map_from_seeds(seeds: dict) -> dict:
    """
    Build a normalized THEMATIQUE_MAP from SEEDS:
    { "Theme name": ["seed1", "seed2", ...], ... } -> { "normalized_term": "Theme name", ... }
    Does not declare or modify SEEDS/THEMATIQUE_MAP here; just returns the map.
    """
    themap = {}
    for theme, terms in (seeds or {}).items():
        for term in terms:
            for v in variants(term):
                themap[normalize_text(v)] = theme
    return themap
THEMATIQUE_MAP = build_thematique_map_from_seeds(SEEDS)


# -----------------------
# Extraction arrondissement / budget / thématique
# -----------------------

def normalize_arrondissement_from_text(text: str) -> Optional[int]:
    txt = normalize_text(text)

    # Regex for "7e arrondissement", "arr 7", "7"
    m = re.search(r"\b(\d{1,2})(?:\s*(?:e|eme|ème|ieme|iéme))?\s*(?:arrondissement|arr|arrond)?\b", txt)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 20:
            return num

    # Fuzzy check for misspelled "arrondissement"
    words = txt.split()
    for w in words:
        close = difflib.get_close_matches(w, ["arrondissement", "arrondissment", "arrond", "arr"], n=1, cutoff=0.75)
        if close:
            # If a number is nearby in the text, extract it
            m2 = re.search(r"\b(\d{1,2})\b", txt)
            if m2:
                num = int(m2.group(1))
                if 1 <= num <= 20:
                    return num

    # Fallback via map textuelle si disponible
    try:
        for k, v in ARRONDISSEMENT_MAP.items():
            if k in txt:
                return v
    except NameError:
        pass

    return None


def parse_budget_from_text(text: str) -> Optional[float]:
    original = "" if text is None else str(text)
    txt = normalize_text(original)
    arr = normalize_arrondissement_from_text(original)

    # Liste des variantes attendues du mot "budget"
    budget_tokens = ["budget", "budjet", "bugdet", "budgét", "budg"]

    # Vérifier si un mot proche de "budget" est présent
    words = txt.split()
    has_budget = any(
        difflib.get_close_matches(w, budget_tokens, n=1, cutoff=0.75)
        for w in words
    )

    # 1) "budget: 400000" / "budget 400k"
    if has_budget:
        mb = re.search(r"(?:budget|budjet|bugdet|budg)(?:\s+de)?\s*:?\s*(\d[\d\s.,]*)(\s*k)?", txt)
        if mb:
            raw = re.sub(r"[^\d]", "", mb.group(1))
            if raw:
                val = int(raw)
                if mb.group(2):
                    val *= 1000
                return float(val)

    # 2) "400 000 €" / "400k eur"
    md = re.search(r"(\d[\d\s.,]*)(\s*k)?\s*(€|eur|euros|\$|usd)", txt)
    if md:
        raw = re.sub(r"[^\d]", "", md.group(1))
        if raw:
            val = int(raw)
            if md.group(2):
                val *= 1000
            return float(val)

    # 3) "400k"
    mk = re.search(r"\b(\d{1,3})\s*k\b", txt)
    if mk:
        return float(int(mk.group(1)) * 1000)

    # 4) Fallback: plus grand nombre significatif, exclut l'arrondissement
    nums = [int(re.sub(r"[^\d]", "", n)) for n in re.findall(r"\b\d[\d\s.,]*\b", original)]
    nums = [n for n in nums if n >= 1000 and n != arr]
    return float(max(nums)) if nums else None

def to_float_budget(b: Any) -> float:
    try:
        if pd.isna(b):
            return np.nan
    except Exception:
        pass
    try:
        return float(str(b).replace(" ", "").replace(",", "."))
    except Exception:
        return np.nan


def map_thematique_free_text(text: str, return_debug: bool = False):
    txt = normalize_text(text)
    scores = {}
    matched = {}
    try:
        keys = sorted(THEMATIQUE_MAP.keys(), key=lambda x: -len(x))
    except NameError:
        keys = []

    # Exact matches first
    for k in keys:
        if k in txt:
            pos = txt.find(k)
            score = 1
            if pos != -1 and pos < len(txt) / 2:
                score += 1
            if len(k.split()) > 1:
                score += 3
            theme = THEMATIQUE_MAP[k]
            scores[theme] = scores.get(theme, 0) + score
            matched.setdefault(theme, []).append((k, score))

    # Fuzzy matching if no exact match
    if not scores and keys:
        words = txt.split()
        for w in words:
            # find close matches among keys
            close = difflib.get_close_matches(w, keys, n=1, cutoff=0.75)
            if close:
                k = close[0]
                theme = THEMATIQUE_MAP[k]
                score = 1
                scores[theme] = scores.get(theme, 0) + score
                matched.setdefault(theme, []).append((k, score))

    if not scores:
        return ("thématique non précisée", []) if return_debug else "thématique non précisée"

    best_theme = max(scores, key=scores.get)
    return (best_theme, matched.get(best_theme, [])) if return_debug else best_theme

# -----------------------
# Préparation du DataFrame
# -----------------------
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Arrondissement
    if "arrondissement_projet_gagnant" in df.columns:
        df["arr_num"] = df["arrondissement_projet_gagnant"].apply(normalize_arrondissement_from_text)
    else:
        df["arr_num"] = None

    # Avancement
    def map_status_db(v):
        txt = normalize_text(v)
        try:
            for k, val in STATUS_MAP.items():  # STATUS_MAP déclaré ailleurs
                if k in txt:
                    return val
        except NameError:
            pass
        return v.upper() if isinstance(v, str) else v

    if "avancement_projet" in df.columns:
        df["avancement_norm"] = df["avancement_projet"].apply(map_status_db)
    else:
        df["avancement_norm"] = None

    # Thématique
    if "thematique" in df.columns:
        df["thematique_norm"] = df["thematique"].apply(map_thematique_free_text)
    else:
        df["thematique_norm"] = "thématique non précisée"

    # Budget numérique
    if "budget_global_projet_gagnant" in df.columns:
        df["budget_num"] = df["budget_global_projet_gagnant"].apply(to_float_budget)
    else:
        df["budget_num"] = np.nan

    # Texte combiné pour scoring simple
    df["text_all"] = (
        df.get("titre_projet_gagnant", pd.Series(dtype=str)).fillna("") + " " +
        df.get("thematique_norm", pd.Series(dtype=str)).fillna("") + " " +
        df.get("avancement_norm", pd.Series(dtype=str)).fillna("")
    )
    return df

def get_nearby_arrondissements(arr: Optional[int]) -> list:
    if arr is None:
        return []
    nearby = []
    if arr > 1:
        nearby.append(arr - 1)
    nearby.append(arr)
    if arr < 20:
        nearby.append(arr + 1)
    return nearby

# -----------------------
# Filtrage FIN + ABANDONNÉ
# -----------------------
def filter_fin_abandoned(df: pd.DataFrame, description: str) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[int], str, Optional[float]]:
    arr_user = normalize_arrondissement_from_text(description)
    theme_user = map_thematique_free_text(description)
    user_budget = parse_budget_from_text(description)

    base_mask = (df["arr_num"] == arr_user) & (df["thematique_norm"] == theme_user)

    df_fin = df[base_mask & (df["avancement_norm"] == "FIN")].copy()
    df_ab = df[base_mask & (df["avancement_norm"].isin(["ABANDON", "ABANDONNÉ", "ABANDONNE"]))].copy()

    return df_fin, df_ab, arr_user, theme_user, user_budget

def filter_projects_with_abandoned(df: pd.DataFrame, description: str):
    return filter_fin_abandoned(df, description)

# -----------------------
# Scoring candidats
# -----------------------
def score_candidates(df_candidates: pd.DataFrame, description: str, user_budget: Optional[float], w_text: float = 0.6, w_budget: float = 0.4) -> pd.DataFrame:
    if df_candidates is None or len(df_candidates) == 0:
        return df_candidates
    desc_norm = normalize_text(description)
    try:
        kws = list(THEMATIQUE_MAP.keys())  # THEMATIQUE_MAP déclaré ailleurs
    except NameError:
        kws = []

    def text_score(row):
        txt = normalize_text(row.get("text_all", ""))
        hits = sum(1 for kw in kws if kw in desc_norm and kw in txt)
        return float(hits)

    def budget_score(row):
        b = row.get("budget_num", np.nan)
        if user_budget is None or pd.isna(b):
            return 0.0
        return 1.0 / (1.0 + abs(b - user_budget) / (user_budget + 1e-6))

    scored = df_candidates.copy()
    scored["score_text"] = scored.apply(text_score, axis=1)
    scored["score_budget"] = scored.apply(budget_score, axis=1)
    scored["score"] = w_text * scored["score_text"] + w_budget * scored["score_budget"]
    return scored.sort_values(by="score", ascending=False)

# -----------------------
# Intent & réponses sociales
# -----------------------

from typing import Iterable, Tuple, Optional

# RapidFuzz is recommended for typo tolerance.
# pip install rapidfuzz
from rapidfuzz import process, fuzz


# Thresholds
SHORT_WORD_THRESHOLD = 75
WORD_THRESHOLD = 80
THEME_THRESHOLD = 70

# Dictionnaire de réponses sociales (personnalisées)
SOCIAL_RESPONSES = {
    "bonjour": [
        "Bonjour 👋 — Ravi de vous voir, parlez-moi de votre projet.",
        "Salut 👋 — Content de vous retrouver, quelle est votre idée aujourd’hui ?"
    ],
    "salut": [
        "Salut 🙂 — Dites-moi votre projet, je vous propose des exemples similaires.",
        "Salut 👋 — Quelle initiative souhaitez-vous explorer ?"
    ],
    "merci": [
        "Avec plaisir 🙏 — Je suis là pour vous aider à trouver des projets similaires.",
        "Merci à vous 🤝 — Voulez-vous que je vous propose des recommandations ?"
    ],
    "au revoir": [
        "Au revoir 👋 — À bientôt pour de nouvelles idées citoyennes.",
        "Bonne journée 🌞 — Revenez quand vous voulez proposer un projet."
    ],
    "ça va": [
        "Ça va très bien 🙂 — Et vous ? Prêt à partager une idée de projet ?",
        "Je vais bien merci 🙌 — Parlez-moi de votre projet pour que je vous propose des suggestions."
    ],
    "bonsoir": [
        "Bonsoir 🌙 — Une idée de projet pour ce soir ?",
        "Bonsoir 👋 — Je peux vous montrer des projets similaires déjà réalisés."
    ]
}

# Liste des mots-clés sociaux
GREETINGS = list(SOCIAL_RESPONSES.keys())

def _normalize_choices(choices: Iterable[str]) -> list:
    return [normalize_text(c) for c in choices]

def fuzzy_contains_any(text: str, choices: Iterable[str],
                       scorer=fuzz.token_sort_ratio, threshold: int = WORD_THRESHOLD
                       ) -> Tuple[Optional[str], int]:
    if not text or not choices:
        return None, 0
    txt_norm = normalize_text(text)
    choices_list = list(choices)
    choices_norm = _normalize_choices(choices_list)
    match = process.extractOne(txt_norm, choices_norm, scorer=scorer)
    if not match:
        return None, 0
    best_norm, score, idx = match
    original_best = choices_list[idx] if 0 <= idx < len(choices_list) else best_norm
    return (original_best, int(score)) if score >= threshold else (None, int(score))

def quick_social_detect(message: str) -> Optional[str]:
    """Retourne le type de salutation détectée (ex: 'bonjour', 'merci')"""
    match, score = fuzzy_contains_any(message, GREETINGS, scorer=fuzz.partial_ratio, threshold=SHORT_WORD_THRESHOLD)
    return match if match else None

def handle_social(message: str) -> str:
    """Retourne une réponse adaptée selon la salutation détectée"""
    greeting = quick_social_detect(message)
    if greeting and greeting in SOCIAL_RESPONSES:
        return random.choice(SOCIAL_RESPONSES[greeting])
    # fallback si pas trouvé
    return "Bonjour 👋 — Parlez-moi de votre projet, je vous propose des projets similaires."

def detect_intent(message: str) -> str:
    """
    Intent detection:
    1) fuzzy social check,
    2) fuzzy 'projet' or thematic keys,
    3) fallback LLM (keeps previous behavior).
    """
    if quick_social_detect(message):
        return "social"

    # Préparer them_keys si disponible
    try:
        them_keys = list(THEMATIQUE_MAP.keys())
    except Exception:
        them_keys = []

    # Fuzzy check for 'projet'
    proj_match, proj_score = fuzzy_contains_any(message, ["projet"], scorer=fuzz.partial_ratio, threshold=WORD_THRESHOLD)
    if proj_match:
        return "project"

    # Fuzzy check for thematic keys
    if them_keys:
        match, score = fuzzy_contains_any(message, them_keys, scorer=fuzz.token_set_ratio, threshold=THEME_THRESHOLD)
        if match:
            return "project"

    # Fallback LLM minimal
    prompt = f'Reponds uniquement par "social" ou "project". Message: "{message}"'
    try:
        reply = ollama_generate(prompt, max_tokens=30, timeout_s=30.0)
        r = (reply or "").lower().strip()
        if "project" in r:
            return "project"
        if "social" in r:
            return "social"
    except Exception:
        pass

    return "project"

# -----------------------
# LLM: appels optimisés
# -----------------------

def _normalize_bullets(raw: str, max_lines: int = 3, pad: bool = True) -> str:
    """
    Nettoie la sortie LLM :
    - supprime préfixes multiples (ex: "- -", "1. -", etc.)
    - assure un préfixe unique "- " par ligne
    - retire les lignes placeholder renvoyées par le LLM (ex: "Suggestion supplémentaire")
    - déduplique les lignes en conservant l'ordre
    - limite à max_lines
    - si pad==True, complète avec des lignes '- Suggestion supplémentaire' si < max_lines
    """
    if not raw:
        return "\n".join(["- Suggestion supplémentaire"] * max_lines) if pad else ""
    # Normalisation des lignes
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    cleaned = []
    for l in lines:
        # Supprime numérotation et tirets en début de ligne
        l = re.sub(r"^\s*(?:\d+[.)]\s*)?[-\s]+", "", l)
        # Compresse espaces internes
        l = re.sub(r"\s+", " ", l).strip()
        if not l:
            continue
        # Filtre les placeholders émis par le modèle (variantes et casse)
        if re.fullmatch(r"(?i)suggestion\s+suppl[ée]mentaire\.?", l):
            continue
        cleaned.append(l)
    # Déduplication en conservant l'ordre
    seen = set()
    deduped = []
    for l in cleaned:
        key = l.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(l)
    # Préfixe unique "- " et coupe à max_lines
    deduped = [f"- {l}" for l in deduped]
    if len(deduped) > max_lines:
        deduped = deduped[:max_lines]
    # Padding uniquement si demandé et nécessaire
    if pad:
        while len(deduped) < max_lines:
            deduped.append("- Suggestion supplémentaire")
    return "\n".join(deduped)


def _normalize_bullets_streaming(raw: str, max_lines: int = 3) -> str:
    """
    Version allégée pour le streaming - ne casse pas le texte incomplet.
    - Gère les lignes partielles sans les filtrer
    - Pas de padding (évite d'afficher des placeholders pendant le streaming)
    - Normalisation minimale pour garder le texte lisible en temps réel
    """
    if not raw:
        return ""
    
    # Enlever les blocs de code markdown si présents
    raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL)
    raw = raw.strip()
    
    if not raw:
        return ""
    
    # Split par lignes mais garde les lignes incomplètes
    lines = raw.split('\n')
    cleaned = []
    
    for i, l in enumerate(lines):
        l = l.strip()
        if not l:
            continue
        
        # Supprime les préfixes multiples mais garde le texte partiel
        l_clean = re.sub(r"^\s*(?:\d+[.)]\s*)?[-•*\s]+", "", l).strip()
        
        # Si la ligne est vide après nettoyage, skip
        if not l_clean:
            continue
        
        # Ne filtre PAS les placeholders pendant le streaming
        # (ils peuvent être du texte partiel en cours de génération)
        
        # Ajoute le préfixe si ce n'est pas déjà une bullet
        if not l.startswith(('-', '•', '*')):
            l_clean = f"- {l_clean}"
        else:
            l_clean = f"- {l_clean}"
        
        cleaned.append(l_clean)
        
        # Limite souple pendant le streaming (permet de voir plus de contenu)
        if len(cleaned) >= max_lines + 1:  # +1 pour voir une ligne en cours
            break
    
    # Déduplication légère seulement sur les lignes complètes
    # (garde les duplicatas potentiels si c'est du texte en cours)
    if len(cleaned) > max_lines:
        seen = set()
        deduped = []
        for l in cleaned[:max_lines]:
            key = l.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(l)
        cleaned = deduped
    
    return '\n'.join(cleaned)
def ollama_generate(prompt: str, max_tokens: int = 80, timeout_s: float = 30.0) -> str:
    """
    Appel rapide à Ollama : stream=False, tokens limités, timeout court.
    Retourne 3 puces max, avec fallback en cas d’erreur.
    """
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.15,
            "top_p": 0.9,
            "stop": ["\n\n"]
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("response") or "").strip()
        if not text:
            return "- Aucune suggestion\n- Réduisez le périmètre ou précisez la thématique\n- Réessayez"
        return _normalize_bullets(text, max_lines=3)
    except Exception as e:
        return f"- Erreur LLM: {e}\n- Essayez encore\n- Ou simplifiez la requête"

def ollama_stream_generate_chunks(prompt: str, max_tokens: int = 80, timeout_s: float = 60.0) -> Generator[str, None, None]:
    """
    Streaming propre: renvoie uniquement le champ 'response' de chaque chunk.
    """
    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.15,
            "top_p": 0.9,
            "stop": ["\n\n"]
        },
    }
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            if "response" in data and data["response"]:
                yield data["response"]
            if data.get("done"):
                break

def warm_up_model_async():
    def _warmup():
        try:
            _ = ollama_generate("Bonjour", max_tokens=16, timeout_s=30.0)
        except Exception:
            pass
    threading.Thread(target=_warmup, daemon=True).start()

# -----------------------
# Prompts de suggestions (orientés projets concrets)
# -----------------------
def build_fast_suggestions_prompt(description: str, df_fin: pd.DataFrame, df_ab: pd.DataFrame, max_items: int = 3) -> str:
    """
    Prompt compact et ciblé: fournit des références mais exige 3 projets NEUFS.
    """
    arr = normalize_arrondissement_from_text(description) or "non précisé"
    budget = parse_budget_from_text(description) or "non précisé"
    theme = map_thematique_free_text(description)

    def row_line(r):
        title = r.get("titre_projet_gagnant", "—")
        bud = r.get("budget_global_projet_gagnant", "—")
        edi = r.get("edition", "—")
        return f"{title} (budget {bud} €, édition {edi})"

    lines = [
        f"Contrainte: arrondissement {arr}, thématique {theme}, budget max {budget} €.",
        "Référence — projets finis (ne pas réutiliser, uniquement contexte):"
    ]
    if df_fin is not None and len(df_fin) > 0:
        for _, r in df_fin.head(max_items).iterrows():
            lines.append(f"- {row_line(r)}")
    else:
        lines.append("- Aucun")

    lines.append("Référence — projets abandonnés (ne pas réutiliser, uniquement contexte):")
    if df_ab is not None and len(df_ab) > 0:
        for _, r in df_ab.head(max_items - 1).iterrows():
            lines.append(f"- {row_line(r)}")
    else:
        lines.append("- Aucun")

    instr = (
    "Propose EXACTEMENT 3 idées de projets citoyens PERMANENTS et INFRASTRUCTURÉS, jamais des événements ponctuels. "
    "Contraintes strictes : "
    "- Chaque idée doit tenir sur une seule ligne et commencer par \"- \". "
    "- Interdiction absolue de proposer des événements, journées, campagnes ponctuelles, ateliers uniques, conférences ou manifestations. "
    "- Chaque idée doit décrire un élément physique ou un service pérenne (ex : aménagement, équipement, installation, local transformé, infrastructure verte, service de prêt, programme permanent). "
    "- Fournir des éléments concrets : emplacement type ou surface, et un indicateur de résultat ou d'usage mesurable (ex : m², %, nombre d'usagers). "
    "- Ne pas citer ni reformuler les projets finis ou abandonnés fournis en référence. "
    "- Rester factuel, précis et réalisable ; éviter le vague et les formulations événementielles. "
    "Donne EXACTEMENT 3 lignes conformes."
)


    return instr + "\n\n" + "\n".join(lines)

# -----------------------
# Clé cache pour suggestions
# -----------------------
def _suggestions_cache_key(description: str, df_fin: pd.DataFrame, df_ab: pd.DataFrame) -> str:
    def digest_df(df: pd.DataFrame, n: int) -> str:
        if df is None or len(df) == 0:
            return "empty"
        rows = []
        for _, r in df.head(n).iterrows():
            rows.append(f"{r.get('titre_projet_gagnant','')}|{r.get('budget_global_projet_gagnant','')}|{r.get('edition','')}")
        return ";".join(rows)
    raw = f"{description}||FIN:{digest_df(df_fin, 3)}||AB:{digest_df(df_ab, 2)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# -----------------------
# Suggestions LLM (direct) et streaming lettre par lettre
# -----------------------
def generate_project_suggestions(description: str, df_fin_scored: pd.DataFrame, df_ab_scored: pd.DataFrame, streaming: bool = False) -> str:
    """
    Non-stream — retourne 3 puces formatées. Cache en mémoire.
    """
    key = _suggestions_cache_key(description, df_fin_scored, df_ab_scored)
    cached = _SUGGESTIONS_CACHE.get(key)
    if cached:
        return cached

    prompt = build_fast_suggestions_prompt(description, df_fin_scored.head(3), df_ab_scored.head(2), max_items=3)
    text = ollama_generate(prompt, max_tokens=250, timeout_s=60.0)
    _SUGGESTIONS_CACHE[key] = text
    return text

def stream_project_suggestions_letters(description: str, df_fin_scored: pd.DataFrame, df_ab_scored: pd.DataFrame) -> Generator[str, None, None]:
    """
    Stream incremental chunks character-by-character for smooth letter-by-letter display.
    - Yields accumulated text after each chunk (creates streaming effect).
    - On completion, yields the final normalized text and caches it.
    """
    key = _suggestions_cache_key(description, df_fin_scored, df_ab_scored)
    cached = _SUGGESTIONS_CACHE.get(key)
    if cached:
        # For cached content, simulate streaming effect
        for i in range(1, len(cached) + 1):
            yield cached[:i]
            time.sleep(0.01)  # Small delay for visual effect
        return
    
    prompt = build_fast_suggestions_prompt(description, df_fin_scored.head(3), df_ab_scored.head(2), max_items=3)
    collected = []
    
    try:
        for chunk in ollama_stream_generate_chunks(prompt, max_tokens=250, timeout_s=45.0):
            if not chunk:
                continue
            collected.append(chunk)
            
            # Yield accumulated text after EVERY chunk (creates letter-by-letter effect)
            partial_raw = "".join(collected)
            # Apply light normalization that preserves partial text
            partial = _normalize_bullets_streaming(partial_raw, max_lines=3)
            yield partial
            
    except Exception:
        # Fallback synchronous if streaming fails
        text = generate_project_suggestions(description, df_fin_scored, df_ab_scored)
        text = _normalize_bullets(text, max_lines=3)
        _SUGGESTIONS_CACHE[key] = text
        yield text
        return
    
    # Final normalization and cache
    final_raw = "".join(collected).strip()
    if not final_raw:
        text = generate_project_suggestions(description, df_fin_scored, df_ab_scored)
        text = _normalize_bullets(text, max_lines=3)
    else:
        text = _normalize_bullets(final_raw, max_lines=3)
    
    _SUGGESTIONS_CACHE[key] = text
    # Yield final normalized version
    yield text



# -----------------------
# Effet machine à écrire
# -----------------------
def _typewriter(text: str, delay_s: float = 0.01, chunk_size: int = 1) -> Generator[str, None, None]:
    if chunk_size <= 1:
        for ch in text:
            yield ch
            if delay_s > 0:
                time.sleep(delay_s)
    else:
        for i in range(0, len(text), chunk_size):
            yield text[i:i+chunk_size]
            if delay_s > 0:
                time.sleep(delay_s)