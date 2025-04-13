# GenAI-FAISS-RAG-Chain

A Retrieval-Augmented Generation (RAG) chatbot implementation using FAISS vector database and Phi-2 language model.

## Overview

This project implements a RAG-based chatbot that uses FAISS for efficient similarity search and the Phi-2 language model for generating responses. The system processes documents, stores them in a vector database, and uses them to provide context-aware responses to user queries.

## Project Structure

```
GenAI-FAISS-RAG-Chain/
├── app/
│   ├── rag_chatbot.py    # Main chatbot implementation
│   └── vectorizer.py     # Document vectorization logic
├── docs/                 # Document storage for RAG
├── model/               # Language model directory
└── my_vector_db/        # FAISS vector database storage
```

## Prerequisites

1. Python 3.8 or higher
2. Required packages (install using requirements.txt)
3. Phi-2 model in GGUF format
4. Download the Phi-2 GGUF model from HuggingFace Hub: https://huggingface.co/TheBloke/phi-2-GGUF/blob/main/phi-2.Q4_K_M.gguf and place it in the `model` folder

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GenAI-FAISS-RAG-Chain.git
cd GenAI-FAISS-RAG-Chain
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. **Important**: Download the Phi-2 model file:
   - Download the `phi-2.Q4_K_M.gguf` file from HuggingFace
   - Place it in the `model/` directory of the project

## Usage

1. Place your documents in the `docs/` directory
2. Run the vectorizer to process documents:
```bash
python app/vectorizer.py
```
3. Start the RAG chatbot:
```bash
python app/rag_chatbot.py
```

## Using the Chatbot UI

1. After starting the RAG chatbot, open your web browser and navigate to:
```bash
http://localhost:7860/
```

2. The chat interface will appear where you can interact with the chatbot.

## Prompting Guidelines

When interacting with the chatbot:
- Be specific and clear with your questions
- The chatbot will use the context from your documents in the `docs/` folder to provide relevant answers
- You can ask questions about the content in your documents, and the system will retrieve the most relevant information
- For best results, phrase your questions naturally and include key terms that match your document content
- The chatbot can handle both general queries and specific questions about the information in your documents

Example prompts:
- "What are the main attractions in Pune?"
- "Who are famous Personalities from Pune?"
- "Tell me more about [specific topic from your documents]"

## Features

- Document vectorization and storage using FAISS
- Efficient similarity search for relevant context retrieval
- Integration with Phi-2 language model for response generation
- Support for multiple document formats (Markdown, HTML)

## Configuration

The project uses default configurations, but you can modify parameters in the respective Python files:
- `vectorizer.py`: Document processing and embedding settings
- `rag_chatbot.py`: Model parameters and RAG chain configuration

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.