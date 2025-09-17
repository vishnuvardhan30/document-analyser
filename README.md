# document-analyser
📄 Document Analyser

Document Analyser is an intelligent Retrieval-Augmented Generation (RAG) system powered by LangChain, LangGraph, and Hugging Face LLMs.
It allows you to ask natural language questions about a PDF document, automatically retrieves relevant sections, and generates a detailed response using a dual-LLM pipeline.

🚀 Features

🔍 Dual-LLM Architecture

Data Gathering LLM: Interprets the user’s query and identifies key topics.

Writing LLM: Generates a detailed, context-aware response using retrieved document chunks.

📚 PDF Ingestion & Vector Store

Loads PDF documents using LangChain’s PyPDFLoader.

Splits text into chunks and stores embeddings in ChromaDB for semantic search.

🛠️ Retriever Tool

Searches the embedded document for relevant passages.

Returns precise context to the response generator.

🤖 Interactive CLI

Run the agent in your terminal.

Ask any question about the document.

Get a final, well-structured answer.

🗂️ Project Structure
document-analyser/
│── app.py              # Main application (agent, retriever, CLI interface)
│── requirements.txt    # Dependencies
│── .env                # Environment variables (Hugging Face API key, etc.)
│── chroma_db/          # Local ChromaDB persistence
│── <your-document>.pdf # PDF file to analyze

⚙️ Installation

Clone the repository

git clone https://github.com/yourusername/document-analyser.git
cd document-analyser


Create and activate a virtual environment

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


Install dependencies

pip install -r requirements.txt


Set environment variables
Create a .env file in the project root and add your Hugging Face token:

HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key

▶️ Usage

Place your target PDF in the project folder and update its path in app.py:

pdf_path = r"E:\projects\RAG_agent\MCIOT_LAB.pdf"


Run the agent:

python app.py


Ask questions interactively:

=== DUAL-LLM RAG AGENT ===
🤖 Data Gathering LLM: HuggingFaceH4/zephyr-7b-beta
✍️  Writing LLM: HuggingFaceH4/zephyr-7b-beta
------------------------------------------------------------

What is your question: Explain the IoT architecture described in the document.

📦 Requirements

All dependencies are listed in requirements.txt
. Key packages include:

langgraph – for workflow orchestration

langchain + langchain_huggingface – for LLM & embeddings

langchain_community – document loaders and utilities

chromadb + langchain_chroma – vector database for retrieval

sentence-transformers – embeddings (all-MiniLM-L6-v2)

transformers – Hugging Face models

python-dotenv – environment variable management

🔮 Roadmap

 Add support for multiple PDFs

 Build a web UI using Streamlit/Flask

 Add citations in responses with document page references

 Support for image-based PDFs (OCR integration)

🤝 Contributing

Contributions are welcome! Please fork the repo and create a pull request.

📜 License

This project is licensed under the MIT License.
