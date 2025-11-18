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
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from pathlib import Path
import re 
from typing import Optional

# === API Key Setup ===
API_KEY = os.environ.get("OPENAI_API_KEY", "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")
os.environ["OPENAI_API_KEY"] = API_KEY

# === Suppress warnings ===
warnings.filterwarnings("ignore")

# === Initialize CLIP Model with Caching ===
@st.cache_resource
def load_clip_model():
    """Caches the CLIP model and processor."""
    with st.spinner("Loading CLIP Model..."):
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()
    return clip_model, clip_processor

clip_model, clip_processor = load_clip_model()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

# === Core RAG Logic Functions ===

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
    """Processes PDF, extracts embeddings, and creates vector store."""
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
            except Exception as e:
                pass 
    doc.close()
    os.remove(temp_pdf_path)

    embeddings_array = np.array(all_embeddings)
    vector_store = FAISS.from_embeddings(
        text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
        embedding=None,
        metadatas=[doc.metadata for doc in all_docs]
    )
    return all_docs, image_data_store, vector_store

# --- NEW HELPER FUNCTION ---
def extract_page_number(query: str) -> Optional[int]:
    """Extracts a page number from the query using regex."""
    match = re.search(r'page\s*(\d+)|on\s*(\d+)', query.lower())
    if match:
        return int(match.group(1) or match.group(2))
    return None

# --- RETRIEVAL FUNCTION WITH PAGE FILTERING ---
def retrieve_multimodal(query, k=5):
    """
    Retrieves the top k most relevant documents, enforcing page metadata filtering
    if a page number is found in the query.
    """
    if "vector_store" not in st.session_state or "all_docs" not in st.session_state:
        st.error("Please process a PDF first.")
        return []

    query_embedding = embed_text(query)
    all_docs = st.session_state.all_docs
    vector_store = st.session_state.vector_store
    image_data_store = st.session_state.image_data_store
    
    target_page_number = extract_page_number(query)
    
    # 1. Page Filtering Logic
    if target_page_number is not None:
        target_page_index = target_page_number - 1
        
        filtered_docs = [
            doc for doc in all_docs if doc.metadata.get("page") == target_page_index
        ]
        
        if not filtered_docs:
            st.warning(f"No content indexed for Page {target_page_number}. Searching globally.")
            pass 
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

    # 2. Standard Semantic Search (Fallback or Global Search)
    return vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)

# --- Message Builder (Unchanged) ---
def create_multimodal_message(query, retrieved_docs):
    """Builds the final list of content objects for the multimodal LLM call."""
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
            # For debugging context, we display the image directly in the message payload.
            content.append({"type": "text", "text": f"\n[Image from page {doc.metadata['page']}]:\n"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data_store[image_id]}"}
            })

    content.append({"type": "text", "text": "\n\nPlease answer the question based on the provided text and images."})
    return {"role": "user", "content": content}

# === Q&A Interface Sub-Function (Updated to display retrieved images to user) ===
def run_qa_interface():
    """Contains the text input and answer logic, run after processing."""
    st.markdown(f"**Document Loaded:** `{st.session_state.uploaded_file_name}`")
    
    # --- PREDEFINED QUESTIONS ---
    if 'questions_set' not in st.session_state:
        st.session_state.questions_set = True
        st.markdown("**Example Questions:**")
        st.markdown("- **Text-based:** *What is a map and why is that theme important?*")
        st.markdown("- **Image-based:** *What are the four intermediate directions shown on the image on page 10?*")

    user_query = st.text_input("🔍 Enter your question:", placeholder="e.g. What does the chart on page 1 show?")

    if st.button("🧠 Ask"):
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

            # === DISPLAY RETRIEVED IMAGES ALONGSIDE ANSWER ===
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

            # === DISPLAY FINAL ANSWER ===
            st.markdown("### 🧠 Answer:")
            st.markdown(f"<div style='background-color:#fff5cc; padding:15px; border-radius:10px; font-size:16px;'>{answer}</div>", unsafe_allow_html=True)
            
        else:
            st.warning("⚠️ Please enter a question to continue.")


# === Main Streamlit App ===
def main():
    st.set_page_config(page_title="Multimodal RAG for PDF Q&A", layout="wide")

    # --- LOGO AND HEADER ---
    LOGO_PATH = Path("assets") / "straive_logo.png" 
    
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        try:
            found_logo = False
            if os.path.isdir('assets'):
                for f in os.listdir('assets'):
                    if 'logo' in f.lower() and f.lower().endswith('.png'):
                        st.image(str(Path('assets') / f), width=150)
                        found_logo = True
                        break
            
            if not found_logo:
                st.warning("⚠️ Logo file not found at 'assets/straive_logo.png'. **ACTION: Run the app from the project root!**")
        except Exception as e:
            st.error(f"Logo display error: Ensure 'assets' folder exists in the project root.")
            
    with col_title:
        st.title("🤖 Multimodal RAG for PDF Q&A")
        st.subheader("Ask questions from text & images inside your PDF!")
        
    st.markdown("---")

    # Check for API Key
    if not os.environ.get("OPENAI_API_KEY") or "YOUR_COMPANY_KEY" in os.environ.get("OPENAI_API_KEY"):
        st.error("API Key not correctly configured. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    
    # --- Initialize flag outside the button ---
    if 'demo_loaded' not in st.session_state:
        st.session_state.demo_loaded = False
        
    # === MODE SELECTION ===
    mode = st.radio(
        "Select Input Mode:",
        ("Upload Your Documents", "Run Demo (Uses Internal Samples)"),
        horizontal=True
    )

    # 1. UPLOAD MODE 
    if mode == "Upload Your Documents":
        uploaded_file = st.file_uploader("Upload a PDF document:", type="pdf")
        
        # Reset demo state when switching mode or uploading a new file
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

    # 2. DEMO MODE
    elif mode == "Run Demo (Uses Internal Samples)":
        DEMO_ROOT_FOLDER = "demo_documents"
        
        # --- File Loading Button ---
        if st.button("Load Demo PDF", type="primary"):
            st.header("⚡ Processing Demo File")
            
            pdf_path = None
            if os.path.isdir(DEMO_ROOT_FOLDER):
                for root, _, files in os.walk(DEMO_ROOT_FOLDER):
                    for f in files:
                        if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(root, f)):
                            pdf_path = os.path.join(root, f)
                            break
                    if pdf_path: break 
            
            if not pdf_path:
                st.error(f"❌ Error: No PDF file found in '{DEMO_ROOT_FOLDER}' or its subfolders. Cannot run demo.")
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
                st.session_state.demo_loaded = True # SET THE FLAG ONLY ON SUCCESS

            except Exception as e:
                st.error(f"An error occurred loading the demo file: {e}. Check file permissions.")
                st.session_state.demo_loaded = False
                return
    
    # --- Q&A Interface persists based on the flag ---
    if st.session_state.demo_loaded:
        run_qa_interface()
    elif mode == "Run Demo (Uses Internal Samples)":
        st.info("Click 'Load Demo PDF' to process the internal file.")


if __name__ == "__main__":
    main()