import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import torch
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

# Streamlit app setup with custom layout and styling
st.set_page_config(page_title="RAG Information Retrieval System", layout="wide")
st.title("RAG Information Retrieval System")

# Header and description for the app
st.markdown("""
Welcome to the **RAG Information Retrieval System**! 
Upload a PDF document, and I'll help you extract and summarize relevant information from it. 
Simply upload a PDF and ask any questions related to its content.
""")

# Sidebar for better UI experience
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Upload a PDF document using the 'Upload your PDF' button.
2. Enter your query in the input box, and I'll provide a relevant summary based on the content.
3. Click 'Submit' to process the query and view the result.
""")

# File upload for PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# Initialize variables
vectorstore = None
summarizer = None
retriever = None

# Process the uploaded PDF
if uploaded_file is not None:
    try:
        # Create a temporary file and write the uploaded PDF to it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load the PDF document from the temporary file
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # Split the text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter()
        split_texts = text_splitter.split_documents(documents)

        # Use BERT-based embedding model for document embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Create the vectorstore using FAISS
        vectorstore = FAISS.from_documents(split_texts, embeddings)

        # Set up the retriever from the vectorstore
        retriever = vectorstore.as_retriever()

        # Load the summarization model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

        st.success("PDF loaded and ready for query!")

        # Clean up the temporary file
        os.remove(tmp_file_path)

    except Exception as e:
        st.error(f"Error while processing the PDF: {e}")

# Function for query handling
def query_rag_system(user_query):
    if not retriever:
        return "Please upload a PDF first."

    try:
        # Retrieve relevant documents from FAISS
        relevant_docs = retriever.get_relevant_documents(user_query)
        
        if not relevant_docs:
            return "No relevant information found in the document."

        # Concatenate all relevant document texts
        context = " ".join(doc.page_content for doc in relevant_docs)
        
        # Ensure that the context and query do not exceed token limits
        max_token_length = 1024  # BART's token limit
        input_text = f"Question: {user_query}\nContext: {context}"
        
        # Tokenize the combined input text to check length
        tokenized_input = summarizer.tokenizer(input_text, return_tensors='pt')
        
        # If the input exceeds the token limit, split it into chunks
        if len(tokenized_input['input_ids'][0]) > max_token_length:
            context_chunks = [context[i:i + max_token_length] for i in range(0, len(context), max_token_length)]
            
            # Summarize each chunk and concatenate the results
            chunk_summaries = []
            for chunk in context_chunks:
                chunk_input = f"Question: {user_query}\nContext: {chunk}"
                chunk_summary = summarizer(chunk_input, max_length=80, min_length=30, do_sample=False)
                chunk_summaries.append(chunk_summary[0]['summary_text'])
            
            # Combine the summaries into a final response
            summarized_answer = " ".join(chunk_summaries)
        else:
            # Directly summarize the input if it's within the token limit
            summarized_answer = summarizer(input_text, max_length=80, min_length=30, do_sample=False)[0]['summary_text']
        
        # Ensure the summary is concise and exact by limiting word count
        summarized_answer = " ".join(summarized_answer.split()[:100])  # Limit to 50 words for precise output
        
        return summarized_answer

    except Exception as e:
        return f"Error processing your query: {e}"

# Streamlit input for user query with enhanced layout
if vectorstore:
    st.subheader("Ask a Question")
    user_query = st.text_input("Enter your query:")

    # Show the response if the user submits a query
    if user_query:
        with st.spinner("Processing your query..."):
            response = query_rag_system(user_query)
            st.write(f"**Answer:** {response}")

# Footer with additional info
st.markdown("""
---
Made with ❤️ using Streamlit, LangChain, and HuggingFace.  
For more information or to contribute, visit [GitHub](https://github.com/your-repository).
""")
