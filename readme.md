# AI Assistant for Research Blogs using RAG

> Retrieval-Augmented Generation (RAG) assistant that answers questions grounded in technical research blog content, reducing hallucinations and keeping responses aligned with the latest material.

---

### Overview

This project implements an **AI assistant for technical research blogs** using **Retrieval-Augmented Generation (RAG)**.  
Instead of relying solely on an LLM’s parametric memory, the assistant retrieves relevant passages from a research blog and uses them as context, significantly reducing hallucinations and outdated knowledge issues when answering user questions.  

The core of the system is built with **LangChain**, **cloud-based embedding models**, and a **vector database** that stores dense representations of blog content.  
By combining semantic retrieval with generation, the assistant can provide more accurate, source-grounded answers tailored to the specific research blog it has indexed.

---

### Project Context: Research Blog Assistant

Modern research blogs and technical writeups are dense, constantly updated, and often cross-reference multiple concepts and prior posts.  
Reading them end-to-end for every question is inefficient, and relying on a generic LLM can lead to incorrect or out-of-date explanations.

This assistant is designed as a **companion for a specific research blog (or a small set of blogs)**.  
You ingest the blog content, build a vector index over it, and then query the assistant in natural language; the model always sees both your question and the most relevant blog passages before generating a response.

---

### Key Features

- **RAG pipeline with LangChain**  
  Orchestrates document ingestion, chunking, embedding, retrieval, and answer generation through a modular LangChain pipeline.

- **Bring your own sources**  
  Paste research blog URLs or upload PDF papers directly into the Streamlit UI. The app ingests both formats with LangChain loaders, HuggingFace embeddings, and a Chroma vector store—the same stack that powered the original onboarding agent.

- **Cloud-based embeddings for long chunks (~1K+ tokens)**  
  Uses hosted embedding models via API to generate dense vector representations for relatively large text chunks, capturing rich context from research-style paragraphs and sections.

- **Vector database–backed semantic retrieval**  
  Stores embeddings in a vector database (e.g., Chroma, Pinecone, Qdrant, or any LangChain-compatible store) to enable fast, similarity-based lookup of the most relevant blog segments.

- **Grounded answers with source context**  
  For each query, the system retrieves top-k relevant chunks from the blog and feeds them to the LLM so that answers are grounded in the original text.

- **Extensible design**  
  You can easily swap out embedding models, vector stores, or LLM providers without changing the high-level flow.

---

### Using the App

1. **Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```
2. **Run the Streamlit interface**
  ```bash
  streamlit run app.py
  ```
3. **Populate the knowledge base**
  - Paste one URL per line for the research blogs you want indexed.
  - Upload any supporting research paper PDFs.
  - Click **Build Knowledge Base**. The app chunks, embeds, and stores everything in Chroma using HuggingFace sentence transformers, mirroring the tooling from the onboarding agent.
4. **Ask questions**
  - Once indexing finishes, the chat input unlocks.
  - Your prompts are answered by a Groq-hosted Llama 3.1 model with retrieval-augmented context from the sources you supplied.

Use **Reset Conversation** to clear the dialogue while keeping the currently indexed sources in memory.

---

### How It Works

At a high level, the system follows the standard RAG pattern, customized for research blogs:

1. **Document ingestion**  
   - Collect blog posts or research articles as Markdown, HTML, or plain text.  
   - Clean titles, headers, and boilerplate to keep only meaningful content.

2. **Chunking & preprocessing**  
   - Split each article into semantically coherent chunks (e.g., paragraph or section-based) sized around **~1K tokens** so the model sees enough local context.  
   - Optionally attach metadata such as blog title, URL, publication date, and section headers.

3. **Embedding & indexing**  
   - Use a **cloud embedding model** (via API) to convert each chunk into a dense vector.  
   - Persist these vectors plus metadata into a **vector database** for efficient similarity search.

4. **Query-time retrieval**  
   - When a user asks a question, embed the query with the same embedding model.  
   - Retrieve the top-k most similar chunks from the vector store as contextual evidence.

5. **Answer generation with RAG**  
   - Construct a prompt that includes: the user’s question, retrieved blog snippets, and high-level instructions to stay faithful to the sources.  
   - Call the LLM to generate a response that cites and summarizes the relevant blog content rather than hallucinating.

---

### Tech Stack

- **Language & Framework**
  - Python (RAG pipeline and orchestration)
  - [LangChain](https://python.langchain.com/) for chains, retrievers, and prompt management

- **Models & Retrieval**
  - Cloud-based **embedding model API** (Hugging face opensource model) to generate vector representations
  - **Vector store / database** for semantic search (any LangChain-compatible backend)

- **Environment & Tooling**
  - `pip` + `virtualenv` or `conda` for dependency management
  - `.env` or environment variables for API keys and credentials

> You can customize the exact providers (LLM, embeddings, vector DB) without changing the overall design.

---

