import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile

# Streamlit app configuration
st.title("PDF Chatbot with Langchain and GPT-4")
st.write("Upload a PDF document and ask questions about its content.")

# Step 1: Upload PDF document
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Step 2: Load PDF document
    def load_pdf(file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents

    # Step 3: Split documents into manageable chunks
    def split_documents(documents):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        return docs

    # Step 4: Create a vector store (FAISS)
    def create_vector_store(docs, embeddings):
        return FAISS.from_documents(docs, embeddings)

    # Step 5: Query the document using LangChain's RetrievalQA
    def query_document(qa_chain, query):
        response = qa_chain.run(query)
        return response

    # Load and process the PDF
    with st.spinner("Processing the PDF..."):
        pdf_documents = load_pdf(temp_file_path)
        splitted_docs = split_documents(pdf_documents)

    # Initialize OpenAI LLM (ChatGPT with GPT-4)
    openai_api_key = "" # Store API key in Streamlit secrets for security
    openai_llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

    # Embeddings and vector store creation
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = create_vector_store(splitted_docs, embeddings)

    # Step 6: Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=openai_llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Step 7: Create a chat input box in Streamlit
    st.write("Now, you can ask questions about the PDF.")
    user_query = st.text_input("Ask a question about the document:")
    
    if user_query:
        with st.spinner("Getting the answer..."):
            response = query_document(qa_chain, user_query)
        st.write(f"**Answer**: {response}")
