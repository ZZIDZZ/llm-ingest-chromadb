# llm-ingest-chromadb

Extract and Ingest Documents of Multiple Formats to ChromaDB

## Overview

`llm-ingest-chromadb` is a Python-based tool designed to extract and ingest documents in various formats into [ChromaDB](https://www.trychroma.com/), a vector database optimized for managing embeddings. This tool is particularly useful for applications involving large language models (LLMs) where efficient retrieval of document embeddings is crucial.

## Features

- **Multi-Format Support**: Supports document extraction from `pdf`, `xls/xlsx`, `doc/docx`, and `ppt/pptx` formats.
- **ChromaDB Integration**: Ingests extracted text into ChromaDB for efficient embedding storage and retrieval.
- **Retrieval Demonstrations**: Provides examples of querying ChromaDB to utilize stored embeddings.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/ZZIDZZ/llm-ingest-chromadb.git
   cd llm-ingest-chromadb
   ```

2. **Create and Activate a Virtual Environment** (optional but recommended):

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Ingesting Documents

1. **Prepare Your Documents**: Place the documents you want to ingest into the `docs-data` directory.

2. **Run the Ingestion Script**:

   ```bash
   python ingest.py
   ```

   This script processes the documents in `docs-data` and ingests their content into ChromaDB.

### Retrieving Embeddings

To demonstrate retrieval operations:

```bash
python retrieval_demo.py
```

This script showcases how to query ChromaDB and retrieve stored embeddings for further analysis or use.

## Project Structure

```
llm-ingest-chromadb/
│
├── docs-data/
│   └── [Your documents here]
│
├── storage/
│   └── [Stores .gguf models for demo]
│
├── utils/
│   ├── __init__.py
│   └── [Utility modules]
│
├── ingest.py
├── retrieval_demo.py
├── requirements.txt
└── README.md
```

- `docs-data/`: Directory for storing documents to be ingested.
- `storage/`: Stores `.gguf` models for demonstration purposes.
- `utils/`: Utility modules for ingestion and retrieval operations.
- `ingest.py`: Script for extracting and ingesting documents into ChromaDB.
- `retrieval_demo.py`: Demonstration script for retrieving embeddings from ChromaDB.
- `requirements.txt`: Lists Python dependencies.

## Dependencies

- Python 3.x
- [ChromaDB](https://www.trychroma.com/)
- Additional dependencies specified in `requirements.txt`

## Contributing

Contributions are welcome! Feel free to fork the repository, submit issues, or open a pull request with your enhancements.

## License

This project is licensed under the GPL 3 License. See the `LICENSE` file for details.
