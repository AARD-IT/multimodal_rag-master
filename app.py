import os
import io
import base64
import fitz 
import torch
import numpy as np
import warnings
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pathlib import Path
import re 
from typing import Optional

# === 1. PAGE CONFIG MUST BE FIRST ===
st.set_page_config(page_title="Analytics Avenue - Multimodal RAG", layout="wide")

# === API Key Setup ===
if "OPENAI_API_KEY" in st.secrets:
    API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    API_KEY = os.environ.get("OPENAI_API_KEY")

# === Suppress warnings ===
warnings.filterwarnings("ignore")

# === Initialize CLIP Model with Caching ===
@st.cache_resource
def load_clip_model():
    with st.spinner("Loading CLIP Model..."):
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
    return clip_model, clip_processor

clip_model, clip_processor = load_clip_model()

if API_KEY:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
else:
    client = None

# =============================================================================
# Core RAG Logic Functions — UNCHANGED
# =============================================================================

def embed_image(image_data):
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

@st.cache_data(show_spinner=False)
def process_pdf(file_bytes, file_name):
    temp_pdf_path = f"./temp_pdf_{file_name}"
    with open(temp_pdf_path, "wb") as f:
        f.write(file_bytes)

    doc = fitz.open(temp_pdf_path)
    all_docs = []
    all_embeddings = []
    image_data_store = {}
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
            text_chunks = splitter.split_documents([temp_doc])
            for chunk in text_chunks:
                all_embeddings.append(embed_text(chunk.page_content))
                all_docs.append(chunk)

        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_id = f"page_{i}_img_{img_index}"

                buffered = io.BytesIO()
                pil_image.save(buffered, format="PNG")
                image_data_store[image_id] = base64.b64encode(buffered.getvalue()).decode()

                all_embeddings.append(embed_image(pil_image))
                all_docs.append(Document(page_content=f"[Image: {image_id}]", metadata={"page": i, "type": "image", "image_id": image_id}))
            except Exception:
                pass

    doc.close()
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)

    embeddings_array = np.array(all_embeddings)
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
        embedding=None,
        metadatas=[doc.metadata for doc in all_docs]
    )
    return all_docs, image_data_store, vector_store

def extract_page_number(query: str) -> Optional[int]:
    match = re.search(r'page\s*(\d+)|on\s*(\d+)', query.lower())
    if match:
        return int(match.group(1) or match.group(2))
    return None

def retrieve_multimodal(query, k=5):
    if "vector_store" not in st.session_state or "all_docs" not in st.session_state:
        st.error("Please process a PDF first.")
        return []

    query_embedding = embed_text(query)
    all_docs = st.session_state.all_docs
    vector_store = st.session_state.vector_store
    image_data_store = st.session_state.image_data_store

    target_page_number = extract_page_number(query)

    if target_page_number is not None:
        target_page_index = target_page_number - 1
        filtered_docs = [doc for doc in all_docs if doc.metadata.get("page") == target_page_index]

        if not filtered_docs:
            st.warning(f"No content indexed for Page {target_page_number}. Searching globally.")
        else:
            filtered_embeddings = [
                embed_text(doc.page_content) if doc.metadata['type'] == 'text'
                else embed_image(Image.open(io.BytesIO(base64.b64decode(image_data_store[doc.metadata['image_id']]))))
                for doc in filtered_docs
            ]
            temp_vector_store = FAISS.from_embeddings(
                text_embeddings=[(doc.page_content, emb) for doc, emb in zip(filtered_docs, filtered_embeddings)],
                embedding=None,
                metadatas=[doc.metadata for doc in filtered_docs]
            )
            return temp_vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)

    return vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)

def create_multimodal_message(query, retrieved_docs):
    content = []
    content.append({"type": "text", "text": f"Question: {query}\n\nContext:"})

    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
    image_data_store = st.session_state.image_data_store

    if text_docs:
        text_context = "\n\n".join([f"[Page {doc.metadata['page']}]: {doc.page_content}" for doc in text_docs])
        content.append({"type": "text", "text": f"Text excerpts:\n{text_context}\n"})

    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id in image_data_store:
            content.append({"type": "text", "text": f"\n[Image from page {doc.metadata['page']}]:\n"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data_store[image_id]}"}
            })

    content.append({"type": "text", "text": "\n\nPlease answer the question based on the provided text and images."})
    return {"role": "user", "content": content}

def run_qa_interface():
    st.markdown(f"**Document Loaded:** `{st.session_state.uploaded_file_name}`")

    if 'questions_set' not in st.session_state:
        st.session_state.questions_set = True
        st.markdown("**Example Questions:**")
        st.markdown("- **Text-based:** *What is a map and why is that theme important?*")
        st.markdown("- **Image-based:** *What are the four intermediate directions shown on the image on page 10?*")

    user_query = st.text_input("🔍 Enter your question:", placeholder="e.g. What does the chart on page 1 show?")

    if st.button("🧠 Ask"):
        if not client:
            st.error("API Key missing. Please add it to Streamlit Secrets.")
            return

        if user_query.strip():
            with st.spinner("🔎 Analyzing document..."):
                results = retrieve_multimodal(user_query, k=5)
                message = create_multimodal_message(user_query, results)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[message],
                    max_tokens=1024
                )
                answer = response.choices[0].message.content

            image_docs = [doc for doc in results if doc.metadata.get("type") == "image"]

            if image_docs:
                st.markdown("### 🖼️ Context Image Retrieved:")
                cols = st.columns(len(image_docs))
                for i, doc in enumerate(image_docs):
                    image_id = doc.metadata.get("image_id")
                    if image_id in st.session_state.image_data_store:
                        image_data = st.session_state.image_data_store[image_id]
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        with cols[i]:
                            st.image(image, caption=f"Page {doc.metadata['page']+1} Context Image", use_container_width=True)

            st.markdown("### 🧠 Answer:")
            st.markdown(f"<div style='background-color:#fff5cc; padding:15px; border-radius:10px; font-size:16px;'>{answer}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter a question to continue.")


# =============================================================================
# MAIN APP
# =============================================================================
def main():

    # ── GLOBAL CSS ───────────────────────────────────────────
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }

        .block-container {
            padding-top: 2rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
            max-width: 100% !important;
        }

        #MainMenu, footer, header { visibility: hidden; }

        /* Brand */
        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 18px;
            margin-bottom: 28px;
        }
        .brand-name {
            font-size: 26px;
            font-weight: 800;
            color: #064b86;
            line-height: 1.3;
        }
        .divider {
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 0 0 32px 0;
        }

        /* Page title */
        h1 {
            font-size: 48px !important;
            font-weight: 900 !important;
            color: #0a0a0a !important;
            letter-spacing: -1px !important;
            line-height: 1.1 !important;
            margin-bottom: 6px !important;
        }

        /* Subtitle */
        .subtitle {
            font-size: 17px;
            font-weight: 500;
            color: #555;
            margin-bottom: 36px;
        }

        /* Headings */
        h2 {
            font-size: 30px !important;
            font-weight: 800 !important;
            color: #0a0a0a !important;
            margin-bottom: 16px !important;
        }
        h3 {
            font-size: 22px !important;
            font-weight: 700 !important;
            color: #0a0a0a !important;
            margin-bottom: 12px !important;
        }

        /* Overview cards */
        .card {
            background: #fff;
            border: 1.5px solid #e5e7eb;
            border-radius: 10px;
            padding: 24px 28px;
            margin-bottom: 20px;
        }
        .card-label {
            font-size: 13px;
            font-weight: 700;
            color: #064b86;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .card-text {
            font-size: 16px;
            font-weight: 500;
            color: #222;
            line-height: 1.7;
        }
        .card ul {
            margin: 0;
            padding-left: 18px;
        }
        .card ul li {
            font-size: 15px;
            font-weight: 500;
            color: #333;
            margin-bottom: 6px;
            line-height: 1.6;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0px;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 32px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px !important;
            font-weight: 600 !important;
            color: #555 !important;
            padding: 12px 28px !important;
            border: none !important;
            background: transparent !important;
        }
        .stTabs [aria-selected="true"] {
            color: #064b86 !important;
            font-weight: 800 !important;
            border-bottom: 3px solid #064b86 !important;
        }

        /* Form labels */
        .stTextInput label,
        .stSelectbox label,
        .stFileUploader label,
        .stRadio label {
            font-size: 15px !important;
            font-weight: 700 !important;
            color: #0a0a0a !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #064b86 !important;
            color: #fff !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            padding: 12px 32px !important;
            border-radius: 6px !important;
            border: none !important;
        }
        .stButton > button:hover {
            background-color: #053d70 !important;
        }

        /* Expander */
        .streamlit-expanderHeader p {
            font-size: 16px !important;
            font-weight: 700 !important;
            color: #0a0a0a !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── BRAND HEADER ─────────────────────────────────────────
    logo_url = "https://raw.githubusercontent.com/Analytics-Avenue/streamlit-dataapp/main/logo.png"
    st.markdown(f"""
    <div class="brand-wrap">
        <img src="{logo_url}" width="64" style="border-radius:8px;">
        <div class="brand-name">
            Analytics Avenue &amp;<br>Advanced Analytics
        </div>
    </div>
    <hr class="divider">
    """, unsafe_allow_html=True)

    # ── PAGE TITLE ───────────────────────────────────────────
    st.title("🤖 Multimodal RAG for PDF Q&A")
    st.markdown('<p class="subtitle">Ask questions from text &amp; images inside your PDF using Generative AI</p>', unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["Overview", "Application"])

    # ════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════
    with tab1:
        st.header("Overview")

        st.markdown("""
        <div class="card">
            <div class="card-label">Purpose</div>
            <div class="card-text">
                Enable intelligent question-answering over PDF documents by combining CLIP-based multimodal embeddings
                with GPT-4o vision — allowing users to query both text content and images embedded within PDFs
                through a unified semantic search and retrieval pipeline.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("Capabilities")
            st.markdown("""
            <div class="card">
                <ul>
                    <li>Processes PDF documents to extract both text chunks and embedded images.</li>
                    <li>Uses CLIP (ViT-B/32) to generate unified embeddings for text and images.</li>
                    <li>FAISS vector store enables fast semantic similarity search across the document.</li>
                    <li>Page-level filtering — query specific pages by mentioning the page number.</li>
                    <li>GPT-4o vision model answers questions using retrieved text and image context.</li>
                    <li>Supports upload mode and built-in demo mode with internal sample PDFs.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("Business Impact")
            st.markdown("""
            <div class="card">
                <ul>
                    <li>Eliminate manual PDF scanning — get instant answers from large documents.</li>
                    <li>Extract insights from charts, diagrams, and images that text search misses.</li>
                    <li>Accelerate document review for research, compliance, and reporting workflows.</li>
                    <li>Reduce analyst time spent on information retrieval from dense PDFs.</li>
                    <li>Scalable to any PDF domain — financial reports, manuals, academic papers.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TAB 2 — APPLICATION (original logic, untouched)
    # ════════════════════════════════════════════════════════
    with tab2:

        if not API_KEY:
            st.error("🚨 API Key not detected! Please set 'OPENAI_API_KEY' in your Streamlit Secrets.")
            st.stop()

        if 'demo_loaded' not in st.session_state:
            st.session_state.demo_loaded = False

        mode = st.radio(
            "Select Input Mode:",
            ("Upload Your Documents", "Run Demo (Uses Internal Samples)"),
            horizontal=True
        )

        if mode == "Upload Your Documents":
            uploaded_file = st.file_uploader("Upload a PDF document:", type="pdf")

            if uploaded_file is not None or st.session_state.demo_loaded:
                st.session_state.demo_loaded = False

            if uploaded_file is not None:
                if "vector_store" not in st.session_state or st.session_state.get('uploaded_file_name') != uploaded_file.name:
                    st.session_state.uploaded_file_name = uploaded_file.name
                    with st.spinner(f"Processing {uploaded_file.name}... This may take a moment."):
                        file_bytes = uploaded_file.getbuffer()
                        st.session_state.all_docs, st.session_state.image_data_store, st.session_state.vector_store = process_pdf(file_bytes, uploaded_file.name)
                    st.success("PDF processed successfully!")

                if "vector_store" in st.session_state:
                    run_qa_interface()

        elif mode == "Run Demo (Uses Internal Samples)":
            DEMO_ROOT_FOLDER = "demo_documents"

            if st.button("Load Demo PDF", type="primary"):
                st.header("⚡ Processing Demo File")

                pdf_path = None
                if os.path.isdir(DEMO_ROOT_FOLDER):
                    for root, _, files in os.walk(DEMO_ROOT_FOLDER):
                        for f in files:
                            if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(root, f)):
                                pdf_path = os.path.join(root, f)
                                break
                        if pdf_path:
                            break

                if not pdf_path:
                    st.error(f"❌ Error: No PDF file found in '{DEMO_ROOT_FOLDER}'. Cannot run demo.")
                    st.session_state.demo_loaded = False
                    return

                try:
                    with open(pdf_path, "rb") as f:
                        file_bytes = f.read()
                    file_name = os.path.basename(pdf_path)

                    with st.spinner(f"Processing Demo PDF: {file_name}..."):
                        st.session_state.all_docs, st.session_state.image_data_store, st.session_state.vector_store = process_pdf(file_bytes, file_name)
                    st.success(f"Demo PDF '{file_name}' processed successfully!")

                    st.session_state.uploaded_file_name = file_name
                    st.session_state.demo_loaded = True

                except Exception as e:
                    st.error(f"An error occurred loading the demo file: {e}. Check file permissions.")
                    st.session_state.demo_loaded = False
                    return

        if st.session_state.demo_loaded:
            run_qa_interface()
        elif mode == "Run Demo (Uses Internal Samples)":
            st.info("Click 'Load Demo PDF' to process the internal file.")


if __name__ == "__main__":
    main()
