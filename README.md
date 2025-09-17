# 📄 Document Analyser

**Document Analyser** is an intelligent Retrieval-Augmented Generation (RAG) system powered by **LangChain**, **LangGraph**, and **Hugging Face LLMs**.  

It allows you to ask **natural language questions** about a PDF document, automatically retrieve relevant sections, and generate a detailed response using a **dual-LLM pipeline**.

---

## 🚀 Features

- 🔍 **Dual-LLM Architecture**  
  - **Data Gathering LLM** – interprets the user’s query and identifies key topics.  
  - **Writing LLM** – generates a detailed, context-aware response using retrieved document chunks.  

- 📚 **PDF Ingestion & Vector Store**  
  - Loads PDF documents using **PyPDFLoader**.  
  - Splits text into chunks and stores embeddings in **ChromaDB** for semantic search.  

- 🛠️ **Retriever Tool**  
  - Searches the embedded document for relevant passages.  
  - Returns precise context to the response generator.  

- 🤖 **Interactive CLI**  
  - Run the agent in your terminal.  
  - Ask any question about the document.  
  - Get a final, well-structured answer.  

---

## 🗂️ Project Structure

   document-analyser/
│── app.py # Main application (agent, retriever, CLI interface)
│── requirements.txt # Dependencies
│── .env # Environment variables (Hugging Face API key, etc.)
│── chroma_db/ # Local ChromaDB persistence
│── <your-document>.pdf # PDF file to analyze

---


---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/document-analyser.git
cd document-analyser
```

2.Create and activate a virtual environment

```bash
python -m venv venv
# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate
```

3.Install dependencies

```bash
pip install -r requirements.txt
```

4.Set environment variables

```bash
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
```

▶️ Usage

1. Place your target PDF in the project folder and update its path in app.py:
```bash
pdf_path = r" ......."
```

2.Run the agent:
```bash
python app.py
```

3.Ask questions interactively:
```bash=== DUAL-LLM RAG AGENT ===
🤖 Data Gathering LLM: HuggingFaceH4/zephyr-7b-beta
✍️ Writing LLM: HuggingFaceH4/zephyr-7b-beta
------------------------------------------------------------

What is your question: Explain the IoT architecture described in the document.
```
## 📦 Requirements

All dependencies are listed in [`requirements.txt`](requirements.txt).  
Here are the key packages this project uses:

- ⚙️ **langgraph** – workflow orchestration  
- 🧠 **langchain** + **langchain_huggingface** – LLMs & embeddings  
- 📂 **langchain_community** – document loaders & utilities  
- 🗄️ **chromadb** + **langchain_chroma** – vector database for retrieval  
- 🔎 **sentence-transformers** – text embeddings (`all-MiniLM-L6-v2`)  
- 🤗 **transformers** – Hugging Face model integration  
- 🔐 **python-dotenv** – environment variable management  

---

## 🔮 Roadmap

Planned enhancements for future versions:

- [ ] 📑 Support for multiple PDFs  
- [ ] 🌐 Web UI using **Streamlit/Flask**  
- [ ] 📖 Citations with **page references**  
- [ ] 🖼️ OCR support for scanned/image PDFs  

---

## 🤝 Contributing

Contributions are always welcome! 🎉  

1. Fork this repository  
2. Create a new branch (`feature/your-feature-name`)  
3. Commit your changes  
4. Push the branch and open a Pull Request  

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.  

---

```

