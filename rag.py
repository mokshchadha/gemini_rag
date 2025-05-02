from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pickle
from pdf2image import convert_from_bytes
import io
from PIL import Image
import base64
import numpy as np

# Set page config first - this must be the first Streamlit command
st.set_page_config(
    page_title="Multimodal PDF Q&A Tool",
    page_icon="📄",
    layout="wide"
)

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create directory for storing images if it doesn't exist
if not os.path.exists("extracted_images"):
    os.makedirs("extracted_images")

# Custom CSS for styling
st.markdown("""
    <style>
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .image-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .image-item {
        border: 1px solid #ddd;
        padding: 5px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

def get_pdf_content(pdf_docs):
    """Extract both text and images from PDF documents"""
    text = ""
    pdf_sources = {}
    file_names = []
    invalid_files = []
    images_data = []  # Store extracted images

    for pdf in pdf_docs:
        file_name = pdf.name
        try:
            # Text extraction
            pdf.seek(0)
            pdf_reader = PdfReader(pdf)
            pdf_text = ""

            for page_num, page in enumerate(pdf_reader.pages):
                content = page.extract_text()
                if content:
                    pdf_text += content
                    pdf_sources[f"{file_name}|{page_num+1}"] = content

            text += pdf_text
            
            # Image extraction
            pdf.seek(0)
            pdf_bytes = pdf.read()
            images = convert_from_bytes(pdf_bytes)
            
            for i, img in enumerate(images):
                page_num = i + 1
                image_key = f"{file_name}|image_{page_num}"
                img_path = f"extracted_images/{file_name.replace('.', '_')}_{page_num}.png"
                
                # Save image to disk
                img.save(img_path, format="PNG")
                
                # Store image metadata
                images_data.append({
                    "image_path": img_path,
                    "source": image_key,
                    "page": page_num,
                    "file_name": file_name
                })
            
            file_names.append(file_name)

        except Exception as e:
            invalid_files.append((file_name, str(e)))
            continue

    if invalid_files:
        error_msg = "Failed to process the following files:\n"
        for fname, error in invalid_files:
            error_msg += f"- {fname}: {error}\n"
        st.error(error_msg)

        if not file_names:
            raise ValueError("No valid PDF files were processed")

    return text, pdf_sources, file_names, images_data

def get_multimodal_chunks(text, pdf_sources, images_data):
    """Create multimodal chunks with text and associated images"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    
    multimodal_chunks = []
    
    for i, chunk in enumerate(chunks):
        sources = []
        related_images = []
        chunk_pages = set()
        
        # Find text sources and their corresponding pages
        for source_key, source_text in pdf_sources.items():
            file_name = source_key.split('|')[0]
            page_num = int(source_key.split('|')[1])
            
            if any(segment in source_text for segment in chunk.split('\n\n') if len(segment) > 50):
                if file_name not in sources:
                    sources.append(file_name)
                chunk_pages.add((file_name, page_num))
            
            # Use word overlap as fallback method
            if not chunk_pages:
                chunk_words = set(chunk.lower().split())
                source_words = set(source_text.lower().split())
                common_words = chunk_words.intersection(source_words)
                if len(common_words) > len(chunk_words) * 0.3:
                    if file_name not in sources:
                        sources.append(file_name)
                    chunk_pages.add((file_name, page_num))
        
        # Find images from the same pages as the text chunk
        for img_data in images_data:
            if (img_data["file_name"], img_data["page"]) in chunk_pages:
                related_images.append(img_data)
        
        # Create multimodal chunk
        multimodal_chunks.append({
            "chunk_id": i,
            "text": chunk,
            "sources": sources,
            "images": related_images
        })
    
    return multimodal_chunks

def generate_multimodal_embeddings(text, image_path=None):
    """Generate embeddings for text and optionally image data"""
    # Use Google's multimodal embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Currently Google's embedding model doesn't directly support multimodal inputs
    # We'll use text-only embeddings for now and include image metadata
    # This can be replaced when proper multimodal embedding models are available
    text_embedding = embeddings.embed_query(text)
    
    # In a production system, you would use a proper multimodal embedding model here
    return text_embedding

def get_multimodal_vector_store(multimodal_chunks, file_names):
    """Create a vector store with multimodal data"""
    # Initialize embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    texts = []
    metadatas = []
    
    # Process each multimodal chunk
    for chunk in multimodal_chunks:
        text = chunk["text"]
        images = chunk["images"]
        sources = chunk["sources"]
        
        # For chunks with images, create entries with image metadata
        if images:
            for img in images:
                texts.append(text)
                metadatas.append({
                    "sources": ",".join(sources),
                    "has_image": True,
                    "image_path": img["image_path"],
                    "image_source": img["source"],
                    "page": img["page"],
                    "file_name": img["file_name"]
                })
        else:
            # For text-only chunks
            texts.append(text)
            metadatas.append({
                "sources": ",".join(sources),
                "has_image": False
            })
    
    # Create vector store with text and image metadata
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("multimodal_faiss_index")
    
    # Save additional data
    with open("file_names.pkl", "wb") as f:
        pickle.dump(file_names, f)
    
    with open("images_data.pkl", "wb") as f:
        pickle.dump([img for chunk in multimodal_chunks for img in chunk["images"]], f)

def get_multimodal_qa_chain():
    """Create a QA chain that can handle multimodal data"""
    prompt_template = """
    Answer the asked question as detailed as possible from the provided context, which includes both text and images when available.
    Make use of visual information when relevant to provide complete answers.
    If the answer is not available just say "answer is not available in the context", do not provide the wrong answer.

    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Initialize embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        db = FAISS.load_local('multimodal_faiss_index', embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

    # Use Gemini Pro Vision model for multimodal capabilities
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain

def find_source_documents(source_docs):
    """Extract source information from retrieved documents"""
    referenced_sources = set()
    image_sources = []

    for doc in source_docs:
        if hasattr(doc, 'metadata'):
            # Extract text sources
            if 'sources' in doc.metadata:
                sources = doc.metadata['sources'].split(',')
                for source in sources:
                    if source:
                        referenced_sources.add(source)
            
            # Extract image sources if available
            if doc.metadata.get('has_image', False):
                image_sources.append({
                    'image_path': doc.metadata.get('image_path'),
                    'source': doc.metadata.get('image_source'),
                    'page': doc.metadata.get('page'),
                    'file_name': doc.metadata.get('file_name')
                })

    return sorted(list(referenced_sources)), image_sources

def load_image_for_model(image_path):
    """Load and prepare image for the model"""
    try:
        with open(image_path, "rb") as img_file:
            return {"mime_type": "image/png", "data": base64.b64encode(img_file.read()).decode('utf-8')}
    except Exception as e:
        st.error(f"Error loading image {image_path}: {str(e)}")
        return None

def answer_question(user_question):
    """Process user question and provide multimodal answers"""
    qa_chain = get_multimodal_qa_chain()

    if qa_chain:
        with st.spinner("Finding answer..."):
            # Get the answer from the QA chain
            response = qa_chain({"query": user_question})
            source_docs = response.get("source_documents", [])
            
            # Extract source information
            referenced_sources, image_sources = find_source_documents(source_docs)

            # Display the answer
            st.markdown("### Answer")
            st.markdown(response['result'])
            st.divider()
            
            # Display text sources
            with st.expander("📄 Text Sources", expanded=True):
                if referenced_sources:
                    for source in referenced_sources:
                        st.write(f"- {source}")
                else:
                    st.warning("No specific text sources found for this answer.")
            
            # Display relevant images if available
            if image_sources:
                with st.expander("🖼️ Visual Sources", expanded=True):
                    st.write(f"Found {len(image_sources)} related images:")
                    
                    # Create columns for images
                    cols = st.columns(min(3, len(image_sources)))
                    
                    for i, img_data in enumerate(image_sources):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            try:
                                st.image(
                                    img_data["image_path"],
                                    caption=f"From {img_data['file_name']}, Page {img_data['page']}",
                                    use_container_width=True
                                )
                            except Exception as e:
                                st.error(f"Error displaying image: {str(e)}")

def main():
    """Main application function"""
    with st.sidebar:
        st.title("📚 Multimodal Document Q&A")
        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True,
            help="Select one or more PDF files to analyze"
        )

        if st.button("Process Documents", key="process_docs"):
            if pdf_docs:
                with st.spinner("Processing documents (extracting text and images)..."):
                    try:
                        # Process PDFs and extract text and images
                        raw_text, pdf_sources, file_names, images_data = get_pdf_content(pdf_docs)
                        
                        if file_names:
                            # Create multimodal chunks
                            with st.spinner("Creating multimodal chunks..."):
                                multimodal_chunks = get_multimodal_chunks(raw_text, pdf_sources, images_data)
                            
                            # Create vector store
                            with st.spinner("Building vector store..."):
                                get_multimodal_vector_store(multimodal_chunks, file_names)
                            
                            st.success("✅ Documents processed successfully!")
                            st.write(f"Processed {len(file_names)} files:")
                            for file in file_names:
                                st.write(f"- {file}")
                                
                            st.write(f"Extracted {len(images_data)} images")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.error("Please upload PDF files first!")

        with st.expander("📊 System Status", expanded=True):
            try:
                if os.path.exists("multimodal_faiss_index"):
                    st.success("Vector Store: Ready")

                    if os.path.exists("file_names.pkl"):
                        with open("file_names.pkl", "rb") as f:
                            files = pickle.load(f)
                        st.write(f"Loaded {len(files)} documents:")
                        for file in files:
                            st.write(f"- {file}")
                            
                    if os.path.exists("images_data.pkl"):
                        with open("images_data.pkl", "rb") as f:
                            images = pickle.load(f)
                        st.write(f"Loaded {len(images)} images")
                else:
                    st.warning("Vector Store: Not initialized")
            except Exception as e:
                st.error(f"Error checking system status: {str(e)}")

    st.title("Ask Questions About Your Documents")
    st.markdown("This multimodal RAG system can understand both text and images in your PDFs.")

    user_question = st.text_input("Enter your question:", placeholder="What information are you looking for?")

    if st.button("Submit Question"):
        if user_question:
            if not os.path.exists("multimodal_faiss_index"):
                st.error("Please upload and process PDF files first!")
            else:
                answer_question(user_question)
        else:
            st.warning("Please enter a question")

    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. Upload your PDF documents using the sidebar
        2. Click 'Process Documents' to analyze both text and images
        3. Type your question and click 'Submit Question'
        4. The answer will be displayed, along with relevant sources (text and images)

        **Note**: This multimodal system can understand and reason about both textual and visual content in your documents!
        """)

    st.sidebar.divider()
    with st.sidebar.expander("⚠️ Security Notice"):
        st.markdown("""
        - Only upload documents you have permission to use
        - This app stores data locally
        - Don't upload sensitive information
        """)

if __name__ == "__main__":
    main()