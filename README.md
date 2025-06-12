# 📰 RockyBot: News Research Tool

**Author:** Arsalan Taassob, PhD

---

## 📖 Overview

**RockyBot** is a lightweight news research assistant built using `Streamlit`, `LangChain`, and `Hugging Face Transformers`. It enables users to:

- Load up to three news URLs  
- Parse and process the HTML contents  
- Create embeddings using HuggingFace models  
- Perform question-answering based on the article contents using a local language model  

---

## 🚀 Features

- HTML parsing via `BSHTMLLoader`  
- Text chunking using `RecursiveCharacterTextSplitter`  
- Embedding with `all-MiniLM-L6-v2`  
- FAISS-based vector search  
- Local inference using `sshleifer/tiny-gpt2`  
- Interactive user interface with `Streamlit`  

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/rockybot.git
cd rockybot

