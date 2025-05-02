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

# Set page config first - this must be the first Streamlit command
st.set_page_config(
    page_title="PDF Q&A Tool",
    page_icon="📄",
    layout="wide"
)

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Custom CSS for styling
st.markdown("""
    <style>
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

def get_pdf_text(pdf_docs):
    text = ""
    pdf_sources = {}
    file_names = []
    invalid_files = []

    for pdf in pdf_docs:
        file_name = pdf.name
        try:
            pdf.seek(0)
            pdf_reader = PdfReader(pdf)
            pdf_text = ""

            for page_num, page in enumerate(pdf_reader.pages):
                content = page.extract_text()
                if content:
                    pdf_text += content
                    pdf_sources[f"{file_name}|{page_num+1}"] = content

            text += pdf_text
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

    return text, pdf_sources, file_names

def get_text_chunks(text, pdf_sources):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)

    chunk_sources = []
    for i, chunk in enumerate(chunks):
        sources = []
        for source_key, source_text in pdf_sources.items():
            file_name = source_key.split('|')[0]

            if any(segment in source_text for segment in chunk.split('\n\n') if len(segment) > 50):
                if file_name not in sources:
                    sources.append(file_name)

        if not sources:
            for source_key, source_text in pdf_sources.items():
                file_name = source_key.split('|')[0]
                chunk_words = set(chunk.lower().split())
                source_words = set(source_text.lower().split())
                common_words = chunk_words.intersection(source_words)
                if len(common_words) > len(chunk_words) * 0.3:
                    if file_name not in sources:
                        sources.append(file_name)

        chunk_sources.append({"chunk_id": i, "text": chunk, "sources": sources})

    return chunk_sources

def get_vector_store(chunk_sources, file_names):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    texts = []
    metadatas = []

    for chunk in chunk_sources:
        texts.append(chunk["text"])
        metadatas.append({"sources": ",".join(chunk["sources"])})

    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")

    with open("file_names.pkl", "wb") as f:
        pickle.dump(file_names, f)

def get_qa_chain():
    prompt_template = """
    Answer the asked question as detailed as possible from the provided context, make sure to provide all the details, if the answer is 
    not available just say "answer is not available in the context", do not provide the wrong answer.

    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return None

    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain

def find_source_documents(source_docs):
    referenced_sources = set()

    for doc in source_docs:
        if hasattr(doc, 'metadata') and 'sources' in doc.metadata:
            sources = doc.metadata['sources'].split(',')
            for source in sources:
                if source:
                    referenced_sources.add(source)

    return sorted(list(referenced_sources))

def answer_question(user_question):
    qa_chain = get_qa_chain()

    if qa_chain:
        with st.spinner("Finding answer..."):
            response = qa_chain({"query": user_question})
            source_docs = response.get("source_documents", [])
            referenced_sources = find_source_documents(source_docs)

            with st.expander("📄 Sources", expanded=True):
                if referenced_sources:
                    for source in referenced_sources:
                        st.write(f"- {source}")
                else:
                    st.warning("No specific sources found for this answer.")

            st.divider()
            st.markdown(response['result'])

def main():
    with st.sidebar:
        st.title("📚 Document Q&A")
        pdf_docs = st.file_uploader(
            "Upload your PDF files",
            accept_multiple_files=True,
            help="Select one or more PDF files to analyze"
        )

        if st.button("Process Documents", key="process_docs"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    try:
                        raw_text, pdf_sources, file_names = get_pdf_text(pdf_docs)
                        if file_names:
                            chunk_sources = get_text_chunks(raw_text, pdf_sources)
                            get_vector_store(chunk_sources, file_names)
                            st.success("✅ Documents processed successfully!")
                            st.write(f"Processed {len(file_names)} files:")
                            for file in file_names:
                                st.write(f"- {file}")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.error("Please upload PDF files first!")

        with st.expander("📊 System Status", expanded=True):
            try:
                if os.path.exists("faiss_index"):
                    st.success("Vector Store: Ready")

                    if os.path.exists("file_names.pkl"):
                        with open("file_names.pkl", "rb") as f:
                            files = pickle.load(f)
                        st.write(f"Loaded {len(files)} documents:")
                        for file in files:
                            st.write(f"- {file}")
                else:
                    st.warning("Vector Store: Not initialized")
            except Exception as e:
                st.error(f"Error checking system status: {str(e)}")

    st.title("Ask Questions About Your Documents")

    user_question = st.text_input("Enter your question:", placeholder="What information are you looking for?")

    if st.button("Submit Question"):
        if user_question:
            if not os.path.exists("faiss_index"):
                st.error("Please upload and process PDF files first!")
            else:
                answer_question(user_question)
        else:
            st.warning("Please enter a question")

    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. Upload your PDF documents using the sidebar
        2. Click 'Process Documents' to analyze them
        3. Type your question and click 'Submit Question'
        4. The sources will be displayed at the top, followed by the answer

        **Note**: Make sure to process your documents before asking questions!
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
