
import streamlit as st
import anthropic
import chromadb
import os
import json
import re
from pypdf import PdfReader

st.set_page_config(page_title="Chatbot WiData", page_icon="🤖", layout="wide")

# API Key
if "ANTHROPIC_API_KEY" in st.secrets:
    os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
client = anthropic.Anthropic()

SYSTEM = """
Sei l'assistente virtuale di WiData Srl, startup IoT e smart cities di Sassari.
Rispondi SOLO basandoti sui documenti forniti nel contesto.
Se non hai informazioni sufficienti, dì chiaramente: 'Non ho questa informazione.'
Non inventare mai dati tecnici, prezzi o specifiche.
"""

@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

def chunka_testo(testo, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(testo):
        chunk = testo[start:start+chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def indicizza_pdf(file_bytes, collection_name="widata"):
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    testo = " ".join(p.extract_text() or "" for p in reader.pages)
    chunks = chunka_testo(testo)
    chroma = get_chroma_client()
    try: chroma.delete_collection(collection_name)
    except: pass
    coll = chroma.create_collection(collection_name)
    coll.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])
    return coll, len(chunks)

def cerca_rag(domanda, collection, n=3):
    if collection is None:
        return []
    risultati = collection.query(query_texts=[domanda], n_results=min(n, collection.count()))
    return risultati["documents"][0]

def guardrail_input(testo):
    if len(testo) > 2000:
        return None, "Messaggio troppo lungo"
    pattern_vietati = ["ignore previous instructions", "ignora le istruzioni"]
    if any(p in testo.lower() for p in pattern_vietati):
        return None, "Input non consentito"
    return testo, None

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Impostazioni")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    n_chunks = st.slider("Chunk RAG", 1, 5, 3)
    st.divider()
    uploaded = st.file_uploader("📄 Carica PDF", type="pdf")
    if uploaded:
        with st.spinner("Indicizzando..."):
            coll, n = indicizza_pdf(uploaded.read())
            st.session_state.collection = coll
            st.success(f"✅ {n} chunk indicizzati")
    st.divider()
    if st.button("🗑️ Nuova chat"):
        st.session_state.messages = []
        st.rerun()

# ── Main ─────────────────────────────────────────────────────────
st.title("🤖 Chatbot WiData")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection" not in st.session_state:
    st.session_state.collection = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Scrivi un messaggio..."):
    testo_ok, errore = guardrail_input(prompt)
    if errore:
        st.error(errore)
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG
    chunks = cerca_rag(prompt, st.session_state.collection, n_chunks)
    contesto = "\n\n---\n\n".join(chunks) if chunks else ""

    messaggio_con_rag = prompt
    if contesto:
        messaggio_con_rag = f"Contesto dai documenti:\n\n{contesto}\n\n---\n\nDomanda: {prompt}"

    st.session_state.messages.append({"role": "user", "content": prompt})
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    history.append({"role": "user", "content": messaggio_con_rag})

    with st.chat_message("assistant"):
        risposta_completa = ""
        placeholder = st.empty()
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            temperature=temperature,
            system=SYSTEM,
            messages=history
        ) as stream:
            for text in stream.text_stream:
                risposta_completa += text
                placeholder.markdown(risposta_completa + "▌")
        placeholder.markdown(risposta_completa)

        if chunks:
            with st.expander(f"📄 {len(chunks)} chunk RAG usati"):
                for i, c in enumerate(chunks):
                    st.caption(f"Chunk {i+1}: {c[:200]}...")

    st.session_state.messages.append({"role": "assistant", "content": risposta_completa})
