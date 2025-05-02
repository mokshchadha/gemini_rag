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
    text = ""  # This will be the concatenated text in order
    pdf_sources = {}  # Maps source_key to text content
    file_names = []
    invalid_files = []
    images_data = []  # Store extracted images
    
    for pdf in pdf_docs:
        file_name = pdf.name
        try:
            # Text extraction
            pdf.seek(0)
            pdf_reader = PdfReader(pdf)
            
            for page_num, page in enumerate(pdf_reader.pages):
                content = page.extract_text()
                if content:
                    source_key = f"{file_name}|{page_num+1}"
                    pdf_sources[source_key] = content
                    text += content  # Append to the combined text in order
            
            # Image extraction - remains the same
            # ...
            
            file_names.append(file_name)

        except Exception as e:
            invalid_files.append((file_name, str(e)))
            continue
            
    # ... rest of function remains the same
    
    return text, pdf_sources, file_names, images_data
def get_multimodal_chunks(text, pdf_sources, images_data):
    """Create multimodal chunks with improved text-to-source mapping"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    
    # Track character offsets in the original text
    offset = 0
    chunk_offsets = []
    for chunk in chunks:
        start = text.find(chunk, offset)
        if start == -1:  # Fallback if exact match not found
            start = offset
        end = start + len(chunk)
        chunk_offsets.append((start, end))
        offset = start + 1  # Move past this chunk
    
    # Create mapping of character positions to sources
    char_to_source = {}
    cumulative_len = 0
    for source_key, source_text in pdf_sources.items():
        source_len = len(source_text)
        for i in range(cumulative_len, cumulative_len + source_len):
            char_to_source[i] = source_key
        cumulative_len += source_len
    
    # Associate chunks with sources based on overlap
    multimodal_chunks = []
    for i, ((start, end), chunk) in enumerate(zip(chunk_offsets, chunks)):
        sources = set()
        chunk_pages = set()
        
        # Find which sources overlap with this chunk
        for pos in range(start, end, max(1, (end-start)//10)):  # Sample positions
            if pos in char_to_source:
                source_key = char_to_source[pos]
                file_name = source_key.split('|')[0]
                page_num = int(source_key.split('|')[1])
                sources.add(file_name)
                chunk_pages.add((file_name, page_num))
        
        # Find images from the same pages as the text chunk
        related_images = []
        for img_data in images_data:
            if (img_data["file_name"], img_data["page"]) in chunk_pages:
                related_images.append(img_data)
        
        # Create multimodal chunk
        multimodal_chunks.append({
            "chunk_id": i,
            "text": chunk,
            "sources": list(sources),  # Convert set to list
            "pages": list(chunk_pages),  # Store page information
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
        pages = chunk["pages"]  # New field with page information
        
        # Create a detailed source string with page numbers
        detailed_sources = []
        for file_name, page_num in pages:
            detailed_sources.append(f"{file_name} (p.{page_num})")
        
        source_string = ", ".join(detailed_sources) if detailed_sources else ", ".join(sources)
        
        # For chunks with images, create entries with image metadata
        if images:
            for img in images:
                texts.append(text)
                metadatas.append({
                    "sources": source_string,
                    "source_files": ",".join(sources),  # Keep original source files
                    "source_pages": ",".join([f"{f}|{p}" for f, p in pages]),  # Store page info
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
                "sources": source_string,
                "source_files": ",".join(sources),  # Keep original source files
                "source_pages": ",".join([f"{f}|{p}" for f, p in pages]),  # Store page info
                "has_image": False
            })
    
    # Create vector store with text and image metadata
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("multimodal_faiss_index")
    
    # Save additional data - remains the same
    # ...

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
    detailed_sources = set()  # For sources with page numbers
    image_sources = []

    for doc in source_docs:
        if hasattr(doc, 'metadata'):
            # Extract text sources with page info if available
            if 'sources' in doc.metadata:
                detailed_source_info = doc.metadata['sources']
                if detailed_source_info:
                    for source in detailed_source_info.split(', '):
                        if source:
                            detailed_sources.add(source)
            
            # Extract file sources as fallback
            if 'source_files' in doc.metadata:
                file_sources = doc.metadata['source_files'].split(',')
                for source in file_sources:
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

    # Prioritize detailed sources with page numbers
    final_sources = list(detailed_sources) if detailed_sources else sorted(list(referenced_sources))
    
    return final_sources, image_sources
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