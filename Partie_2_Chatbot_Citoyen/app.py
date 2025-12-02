# app.py
import streamlit as st
import pandas as pd


from utils import (
    load_dataframe_from_sql,
    prepare_dataframe,
    filter_fin_abandoned,
    score_candidates,
    get_nearby_arrondissements,
    detect_intent,
    handle_social,
    stream_project_suggestions_letters,
    warm_up_model_async,
)

# -----------------------
# Compatibilité rerun
# -----------------------
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# -----------------------
# CSS Messenger-style
# -----------------------
CSS = """
<style>
.chat-wrap { display:flex; flex-direction:column; gap:12px; padding:12px; }
.chat-row { display:flex; width:100%; margin-bottom:20px;}
.bubble.user {
  margin-left:auto; background:linear-gradient(135deg,#4f9cff,#2b6ef6);
  color:#fff; padding:12px 16px; border-radius:18px 18px 6px 18px;
  max-width:78%; box-shadow:0 2px 8px rgba(43,110,246,0.18); font-size:14px; line-height:1.4;
}
.bubble.assistant {
  margin-right:auto; background:#f1f3f5; color:#111827;
  padding:12px 16px; border-radius:18px 18px 18px 6px;
  max-width:78%; box-shadow:0 2px 6px rgba(0,0,0,0.06); font-size:14px; line-height:1.4;
}
.bubble .meta { font-size:12px; color:#6b7280; margin-bottom:12px; }
.analysis { background:#fff; border-radius:12px; padding:10px; margin-top:6px; }
.analysis-columns { display:flex; gap:12px; }
.analysis-col { flex:1; background:#fafafa; border-radius:8px; padding:8px; }
.analysis-col h4 { margin:0 0 8px 0; font-size:14px; }
.analysis-item { margin-bottom:8px; font-size:13px; }
.llm-summary { margin-top:10px; font-size:13px; color:#FFFFFF; white-space:pre-line; }
.llm-summary1 { margin-top:10px; font-size:13px; color:#111827; white-space:pre-line; }
</style>
"""

st.set_page_config(page_title="Chatbot Projets Citoyens", page_icon="💬", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; padding:10px;'>
        <h1 style='color:#2b6ef6; font-size:36px; margin-bottom:0;'>
            💡 Donnez Vie à Votre Projet Citoyen et Inspirez Votre Quartier
        </h1>
        <p style='color:#6b7280; font-size:18px; margin-top:4px;'>
            Partagez vos idées, explorez des initiatives déjà réalisées et découvrez des recommandations adaptées 🛠️✨
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Charger DB
# -----------------------
DB_PATH = r"C:\Users\user\Desktop\app recommondation\bp_projects.db"
TABLE_NAME = "projects"

@st.cache_data(ttl=300)
def load_and_prepare(db_path: str, table_name: str) -> pd.DataFrame:
    df = load_dataframe_from_sql(db_path, table_name)
    df = prepare_dataframe(df)
    return df

df = load_and_prepare(DB_PATH, TABLE_NAME)

# -----------------------
# Warmup modèle (une seule fois par session)
# -----------------------
if "model_warmed" not in st.session_state:
    st.session_state.model_warmed = False

if not st.session_state.model_warmed:
    try:
        # lancer le warmup en arrière-plan (best-effort)
        warm_up_model_async(timeout_s=60.0, max_tokens=8)
    except Exception:
        pass
    st.session_state.model_warmed = True

# -----------------------
# Session state
# -----------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------
# Rendu bulle
# -----------------------
def render_message(entry: dict):
    role = entry.get("role", "assistant")
    cls = "user" if role == "user" else "assistant"
    meta = entry.get("meta")
    text = entry.get("text", "")
    df_fin = entry.get("finis")
    df_ab = entry.get("abandonnés")
    suggestions = entry.get("suggestions")

    html = '<div class="chat-row">'
    html += f'<div class="bubble {cls}">'
    if meta:
        html += f'<div class="meta">{meta}</div>'
    html += f"{text}"

    if df_fin is not None or df_ab is not None:
        html += '<div class="analysis"><div class="analysis-columns">'
        # FIN
        html += '<div class="analysis-col"><h4>✅ Projets finis</h4>'
        if isinstance(df_fin, pd.DataFrame) and len(df_fin) > 0:
            for _, r in df_fin.iterrows():
                html += f'<div class="analysis-item"><strong>{r.get("titre_projet_gagnant","—")}</strong><br>'
                html += f'Ce projet a été réalisé dans le {r.get("arrondissement_projet_gagnant","—")}ᵉ arrondissement '
                html += f'avec un budget de {r.get("budget_global_projet_gagnant","—")} €, avancement {r.get("avancement_norm","—")}. '
                html += f'Édition {r.get("edition","—")}.</div>'
        else:
            html += '<div class="analysis-item">Aucun projet fini correspondant.</div>'
        html += '</div>'
        # ABANDONNÉ
        html += '<div class="analysis-col"><h4>⚠️ Projets abandonnés</h4>'
        if isinstance(df_ab, pd.DataFrame) and len(df_ab) > 0:
            for _, r in df_ab.iterrows():
                html += f'<div class="analysis-item"><strong>{r.get("titre_projet_gagnant","—")}</strong><br>'
                html += f'Prévu dans le {r.get("arrondissement_projet_gagnant","—")}ᵉ arrondissement '
                html += f'avec un budget de {r.get("budget_global_projet_gagnant","—")} €, mais abandonné. '
                html += f'Édition {r.get("edition","—")}.</div>'
        else:
            html += '<div class="analysis-item">Aucun projet abandonné correspondant.</div>'
        html += '</div></div>'

        # Suggestions
        if suggestions == "pending":
            html += '<div class="llm-summary1">⏳ Attendez, génération des suggestions...</div>'
        elif isinstance(suggestions, str) and suggestions.strip():
            html += f'<div class="llm-summary1"><strong>💡 Suggestions :</strong><br>{suggestions}</div>'

        html += '</div>'

    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)

# -----------------------
# Input utilisateur
# -----------------------
user_input = st.chat_input("Décris ton idée de projet (arrondissement, budget, thématique)")

# -----------------------
# Traitement
# -----------------------
if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})
    intent = detect_intent(user_input)

    if intent == "social":
        reply = handle_social(user_input)
        st.session_state.chat_history.append({"role": "assistant", "text": reply})
    else:
        df_fin, df_ab, arr_user, theme_user, user_budget = filter_fin_abandoned(df, user_input)
        df_fin_scored = score_candidates(df_fin, user_input, user_budget)
        df_ab_scored = score_candidates(df_ab, user_input, user_budget)

        meta = f"Arrondissement: {arr_user or 'non précisé'} — Thématique: {theme_user or 'non précisée'}"

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "text": "Voici des projets similaires trouvés dans la base",
                "meta": meta,
                "finis": df_fin_scored.head(3),
                "abandonnés": df_ab_scored.head(5),
                "suggestions": "pending",
            }
        )

        # Afficher immédiatement colonnes + placeholder
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for entry in st.session_state.chat_history:
            render_message(entry)
        st.markdown('</div>', unsafe_allow_html=True)

        # Générer suggestions avec effet lettre par lettre
        placeholder = st.empty()
        placeholder.markdown("<div class='llm-summary'>⏳ Génération des suggestions…</div>", unsafe_allow_html=True)

        final_text = ""
        try:
            # Stream lettre par lettre - chaque chunk contient le texte accumulé
            for chunk in stream_project_suggestions_letters(user_input, df_fin_scored, df_ab_scored):
                final_text = chunk
                # Mise à jour en temps réel du placeholder
                placeholder.markdown(
                    f"<div class='llm-summary'><strong>💡 Suggestions :</strong><br>{final_text}</div>", 
                    unsafe_allow_html=True
                )
        except Exception as e:
            final_text = f"- Erreur lors de la génération: {str(e)}"
            placeholder.markdown(
                f"<div class='llm-summary'><strong>💡 Suggestions :</strong><br>{final_text}</div>", 
                unsafe_allow_html=True
            )

        # Vérifier qu'on a du contenu
        if not final_text or final_text.strip() == "":
            final_text = "- Aucune suggestion disponible pour le moment. Réessayez."
            placeholder.markdown(
                f"<div class='llm-summary'><strong>💡 Suggestions :</strong><br>{final_text}</div>", 
                unsafe_allow_html=True
            )

        # Mettre à jour l'historique avec le texte final
        st.session_state.chat_history[-1]["suggestions"] = final_text
        safe_rerun()

# -----------------------
# Affichage historique
# -----------------------
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
for entry in st.session_state.chat_history:
    render_message(entry)
st.markdown('</div>', unsafe_allow_html=True)