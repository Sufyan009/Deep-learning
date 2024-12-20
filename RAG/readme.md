### **RAG Information Retrieval System Overview**

**Purpose:**  
- Retrieve and summarize relevant content from uploaded PDF documents based on user queries.

---

### **Key Technologies:**

1. **Streamlit** – Web app framework for interactive UI and file uploads.
2. **LangChain** – Framework for NLP integration (HuggingFace models, FAISS).
3. **HuggingFace**  
   - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` for document embeddings.
   - **Summarization**: `facebook/bart-large-cnn` for generating answers.
4. **FAISS** – Vector search engine for fast document retrieval.
5. **PyTorch** – Used to run HuggingFace models.
6. **spaCy** – Optional text preprocessing (stopword removal, lemmatization).

---

### **Techniques:**

1. **Information Retrieval** – FAISS for fast document search.
2. **Text Chunking** – Splitting long documents into manageable chunks.
3. **Document Embeddings** – Using BERT-based embeddings for document representation.
4. **Summarization** – Using BART for concise answers.
5. **Token Management** – Ensures inputs do not exceed model token limits.

---

### **Workflow:**

1. **Upload PDF** – User uploads a PDF.
2. **Process Document** – Split into chunks for retrieval.
3. **Submit Query** – User enters a query.
4. **Retrieve & Summarize** – Relevant content is retrieved and summarized.
5. **Display Answer** – Provide a concise, context-aware response.

---

### **Summary:**

- **Technologies**: Streamlit, LangChain, HuggingFace, FAISS, PyTorch
- **Techniques**: Retrieval, Summarization, Embeddings, Chunking
- **Models**: BART, Sentence Transformers
