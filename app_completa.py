import streamlit as st
import anthropic
import os
from pypdf import PdfReader
import io

st.set_page_config(
    page_title="Chatbot WiData",
    page_icon="🤖",
    layout="wide"
)

# API Key — funziona sia su Streamlit Cloud che in locale
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

client = anthropic.Anthropic()

SYSTEM = """
Sei l'assistente virtuale di WiData Srl, startup IoT e smart cities di Sassari.
Rispondi SOLO basandoti sui documenti forniti nel contesto.
Se non hai informazioni sufficienti, dì chiaramente: 'Non ho questa informazione.'
Non inventare mai dati tecnici, prezzi o specifiche.
"""

# ── RAG senza ChromaDB ────────────────────────────────────────────────────────

def chunka_testo(testo, chunk_size=400, overlap=50):
    chunks, start = [], 0
    while start < len(testo):
        chunk = testo[start:start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def indicizza_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    testo = " ".join(p.extract_text() or "" for p in reader.pages)
    return chunka_testo(testo)

def cerca_rag(domanda, chunks, n=3):
    """Ricerca per parole chiave — funziona su qualsiasi versione Python."""
    if not chunks:
        return []
    parole = [p for p in domanda.lower().split() if len(p) > 2]
    scored = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(1 for p in parole if p in chunk_lower)
        scored.append((score, chunk))
    scored.sort(reverse=True)
    return [c for _, c in scored[:n] if _ > 0]

# ── Guardrail input ───────────────────────────────────────────────────────────

def guardrail_input(testo):
    if len(testo) > 2000:
        return None, "Messaggio troppo lungo (max 2000 caratteri)"
    pattern_vietati = [
        "ignore previous instructions",
        "ignora le istruzioni precedenti",
        "forget your system prompt",
        "<script",
    ]
    if any(p in testo.lower() for p in pattern_vietati):
        return None, "Input non consentito"
    return testo, None

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "token_totali" not in st.session_state:
    st.session_state.token_totali = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Impostazioni")

    nome_chatbot = st.text_input("Nome chatbot", "Chatbot WiData")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    n_chunks = st.slider("Chunk RAG", 1, 5, 3)

    st.divider()

    uploaded = st.file_uploader("📄 Carica PDF", type="pdf")
    if uploaded:
        with st.spinner("Indicizzando..."):
            chunks = indicizza_pdf(uploaded.read())
            st.session_state.chunks = chunks
            st.success(f"✅ {len(chunks)} chunk indicizzati")

    st.divider()

    costo = st.session_state.token_totali / 1_000_000 * 1.0
    st.metric("Messaggi", len(st.session_state.messages))
    st.metric("Token usati", st.session_state.token_totali)
    st.metric("Costo stimato", f"${costo:.5f}")

    st.divider()

    if st.button("🗑️ Nuova chat"):
        st.session_state.messages = []
        st.session_state.token_totali = 0
        st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────

st.title(f"🤖 {nome_chatbot}")
st.caption("Assistente virtuale per prodotti IoT e smart cities — WiData Srl")

if not st.session_state.chunks:
    st.info("💡 Carica un PDF dalla sidebar per attivare il RAG. "
            "Senza documento il chatbot risponde con la sua conoscenza generale.")

# Mostra history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input utente
if prompt := st.chat_input("Scrivi un messaggio..."):

    # Guardrail input
    testo_ok, errore = guardrail_input(prompt)
    if errore:
        st.error(errore)
        st.stop()

    # Mostra messaggio utente
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG
    chunks_trovati = cerca_rag(prompt, st.session_state.chunks, n_chunks)
    contesto = "\n\n---\n\n".join(chunks_trovati) if chunks_trovati else ""

    # Costruisci messaggio con contesto RAG
    if contesto:
        messaggio_rag = (
            f"Documenti di riferimento:\n\n{contesto}\n\n"
            f"---\n\nDomanda: {prompt}"
        )
    else:
        messaggio_rag = prompt

    # Aggiorna history — salva il messaggio originale (senza contesto)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Costruisci history per l'API — sostituisci l'ultimo con la versione RAG
    history_api = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]
    history_api.append({"role": "user", "content": messaggio_rag})

    # Genera risposta con streaming
    with st.chat_message("assistant"):
        risposta_completa = ""
        placeholder = st.empty()

        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=700,
            temperature=temperature,
            system=SYSTEM,
            messages=history_api
        ) as stream:
            for text in stream.text_stream:
                risposta_completa += text
                placeholder.markdown(risposta_completa + "▌")

        placeholder.markdown(risposta_completa)

        # Mostra chunk usati
        if chunks_trovati:
            with st.expander(f"📄 {len(chunks_trovati)} chunk RAG usati"):
                for i, c in enumerate(chunks_trovati):
                    st.caption(f"Chunk {i+1}: {c[:200]}...")

        # Feedback
        st.feedback("thumbs")

    # Salva risposta in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": risposta_completa
    })

    # Aggiorna contatore token (stima)
    st.session_state.token_totali += len(risposta_completa) // 4
