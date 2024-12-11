# RAG Information Retrieval System

Welcome to the **RAG Information Retrieval System**! This project is designed to enable information retrieval and summarization from PDF documents using state-of-the-art Retrieval-Augmented Generation (RAG) techniques. The system combines tools such as **LangChain**, **HuggingFace**, and **FAISS** to deliver accurate and concise responses based on user queries.

---

## Features

- **PDF Upload and Processing**: Upload PDF documents for processing and retrieval.
- **Query-Based Information Retrieval**: Extract and summarize relevant content based on user queries.
- **Advanced Embeddings**: Leverages `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
- **Summarization Model**: Utilizes `facebook/bart-large-cnn` for summarizing retrieved content.
- **Streamlit UI**: Provides an intuitive and interactive user interface for seamless experience.

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository.git
   cd your-repository
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## How to Use

1. Launch the application using the command mentioned above.
2. Upload a PDF document via the **Upload your PDF** button.
3. Enter a query in the input box to extract and summarize information from the uploaded PDF.
4. View the summarized response directly on the application.

---

## Project Workflow

1. **PDF Document Loading**: Uploaded PDFs are processed using `PyPDFLoader` to extract text content.
2. **Text Splitting**: Documents are split into smaller chunks using `RecursiveCharacterTextSplitter` for efficient processing.
3. **Embeddings and VectorStore**: 
   - Text chunks are embedded using `HuggingFaceEmbeddings`.
   - The FAISS library stores embeddings for quick and accurate retrieval.
4. **Query Retrieval**:
   - The user's query is matched against the stored embeddings.
   - Relevant document chunks are retrieved.
5. **Summarization**: Retrieved content is summarized using the `facebook/bart-large-cnn` model to deliver concise responses.

---

## Technical Stack

- **Framework**: Streamlit for the user interface.
- **Libraries**:
  - **LangChain** for document processing and vector storage.
  - **HuggingFace Transformers** for embeddings and summarization.
  - **FAISS** for efficient vector-based retrieval.
  - **PyTorch** for leveraging GPU capabilities (if available).

---

## Prerequisites

- Python 3.7 or higher
- Dependencies listed in `requirements.txt`
- GPU support is recommended for optimal performance (optional).

---

## Contributing

We welcome contributions to enhance the functionality and performance of this project. To contribute:
1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request with a detailed description of changes.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

--
