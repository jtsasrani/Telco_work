# Implementation Plan - Real 3GPP RAG Pipeline (rag_pipeline)

We will replace the current mock/static RAG retriever with a real, fully functional 3GPP Specification RAG pipeline. This pipeline will ingest actual 3GPP specifications, chunk them recursively, embed them using `sentence-transformers` (`all-MiniLM-L6-v2`), index them with `FAISS`, and serve them to the fine-tuned model.

As requested, all RAG code will be placed in a standard Python package directory named `rag_pipeline/` (replacing the old `rag/` folder). All imports in the application will be updated.

We are implementing this on a new branch: `feature/real-3gpp-rag`.

---

## User Review Required

> [!IMPORTANT]
> **Dependencies on the AMD Server**:
> The real RAG pipeline requires the following Python libraries installed on the serving node:
> - `sentence-transformers` (for generating L6 embeddings)
> - `faiss-cpu` or `faiss-gpu` (for fast vector similarity search)
> - `pypdf` (pure-Python PDF extraction library, easy to install and lightweight)
> - `requests` (already present, used for downloading zipped specs)
> 
> You can install them by running:
> ```bash
> pip install sentence-transformers faiss-cpu pypdf
> ```
> 
> **3GPP Specification Ingestion Strategy**:
> Downloading *all* 3GPP specifications (hundreds of thousands of files across decades of GSM/UMTS/LTE/5G) is inefficient, takes massive disk space, and adds noise. Instead, we will index the **10 most crucial active 5G (Release 15/16/17/18) core, RAN, and security specifications** representing the actual domain of our fine-tuned diagnostics model:
> - **TS 38.331** (5G NR RRC protocol)
> - **TS 23.501** (5G System Architecture)
> - **TS 24.501** (5G Non-Access-Stratum NAS protocol)
> - **TS 38.300** (5G NR Overall Description)
> - **TS 38.401** (5G NG-RAN Architecture)
> - **TS 24.229** (IMS Call Control / SIP / VoNR)
> - **TS 38.213** (Physical Layer Control)
> - **TS 38.214** (Physical Layer Data)
> - **TS 38.321** (MAC protocol)
> - **TS 38.304** (User Equipment UE procedures in Idle mode)
> 
> We will provide a script to download these directly from the official 3GPP server. You can also manually drop any other `.pdf`, `.docx`, or `.txt` specification files into the `/workspace/3gpp_docs` folder and re-run ingestion to index them.

---

## Open Questions

> [!NOTE]
> None at the moment. The plan covers moving code to the new `rag_pipeline` directory, the curated 3GPP file downloader, and the dual fallback vector retrieval engine.

---

## Proposed Changes

### Ingestion & Data Acquisition

#### [NEW] [download_specs.py](file:///c:/Users/jittu/AMD%20Hackathon/rag_pipeline/download_specs.py)
A helper utility to download zipped 3GPP specs directly from the official 3GPP FTP/HTTP servers and extract them to the raw documents folder.

- Downloads the curated 5G specification sheets listed above.
- Automatically handles extraction of compressed Word (`.docx`) files from zip archives.

#### [NEW] [ingest.py](file:///c:/Users/jittu/AMD%20Hackathon/rag_pipeline/ingest.py)
An ingestion pipeline script to process raw specification documents and compile a vector search index.

- Scans the `3gpp_docs` directory for `.pdf`, `.docx`, and `.txt` files.
- Extracts text content:
  - `.txt` read directly.
  - `.pdf` parsed using `pypdf`.
  - `.docx` parsed using a lightweight pure-Python ZIP and XML extractor (`word/document.xml`), ensuring zero external system dependencies.
- Recursively chunks text (default size 800 characters with 100 character overlap).
- Generates 384-dimensional embeddings using Hugging Face's `all-MiniLM-L6-v2`.
- Builds a FAISS index.
- Serializes and saves the index to `rag_pipeline/index/index.faiss` and metadata to `rag_pipeline/index/metadata.pkl`.

---

### Retrieving & Serving Architecture

#### [NEW] [spec_retriever.py](file:///c:/Users/jittu/AMD%20Hackathon/rag_pipeline/spec_retriever.py)
Upgrade the retriever module to support loading the serialized FAISS index from disk.

- Checks for the existence of `rag_pipeline/index/index.faiss` and `rag_pipeline/index/metadata.pkl`.
- If they exist and imports succeed, it sets `use_embeddings = True` and loads the model and FAISS index.
- Otherwise, logs a warning and falls back to TF-IDF on the static knowledge base.
- Standardizes search output formats so that both embedding and TF-IDF search return identical dictionary keys.

#### [DELETE] [spec_retriever.py](file:///c:/Users/jittu/AMD%20Hackathon/rag/spec_retriever.py)
Delete the old `rag` directory.

#### [MODIFY] [fused_server.py](file:///c:/Users/jittu/AMD%20Hackathon/fused_server.py)
Enable auto-RAG loading and prevent double-prompt-augmentation.

- Change import from `rag.spec_retriever` to `rag_pipeline.spec_retriever`.
- Change `SpecRetriever(use_embeddings=False)` to `SpecRetriever()`, relying on the new auto-detect behavior.
- Add an optimization to the `chat_completions` endpoint: if the client has already performed RAG and injected `--- RETRIEVED 3GPP SPECIFICATION CONTEXT ---` into the system prompt, skip server-side RAG to prevent redundant API queries.

#### [MODIFY] [app_v3_decoupled.py](file:///c:/Users/jittu/AMD%20Hackathon/app_v3_decoupled.py)
Upgrade client-side RAG to utilize the auto-detecting `SpecRetriever` from the new package.

- Change import from `rag.spec_retriever` to `rag_pipeline.spec_retriever`.
- Change `SpecRetriever(use_embeddings=False)` to `SpecRetriever()` to allow loading vector indices if the user builds the index locally on Windows, while gracefully falling back to TF-IDF if the dependencies are missing.

---

## Verification Plan

### Automated Tests
1. **Syntax Check**: Run bytecode compilation on all changed files:
   ```powershell
   python -m py_compile rag_pipeline/download_specs.py
   python -m py_compile rag_pipeline/ingest.py
   python -m py_compile rag_pipeline/spec_retriever.py
   python -m py_compile fused_server.py
   python -m py_compile app_v3_decoupled.py
   ```
2. **Retriever Unit Test**: Run `python rag_pipeline/spec_retriever.py` to verify the self-test execution (runs TF-IDF and outputs correct scores).

### Manual Verification
1. Run `python rag_pipeline/download_specs.py` to verify downloading and extracting a small 3GPP specification.
2. Run `python rag_pipeline/ingest.py` (with `--quick` or small documents) to verify that PDF/DOCX/TXT text extraction, chunking, embedding, and FAISS indexing execute without errors.
3. Verify that `fused_server.py` boots successfully and outputs either loading FAISS index or falling back gracefully.
4. Verify that sending a message in `app_v3_decoupled.py` displays retrieved references in the client UI collapsible card.
