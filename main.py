import os
import pickle
import time
import requests
import streamlit as st

from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_community.chains import RetrievalQAWithSourcesChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline



model_id = "sshleifer/tiny-gpt2"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )

    llm = HuggingFacePipeline(pipeline=pipe)

except Exception as e:
    st.error(f"❌ Failed to load the model: {e}")
    st.stop()


def load_url_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("temp.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        loader = BSHTMLLoader("temp.html")
        return loader.load()
    else:
        raise ValueError(f"Failed to load {url}, status code: {response.status_code}")

st.title("📰 RockyBot: News Research Tool 📈")
st.sidebar.title("Enter News URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    data2 = []
    for url in urls:
        if url.strip():
            try:
                data2.extend(load_url_content(url))
            except Exception as e:
                st.warning(f"⚠️ Could not load {url}: {e}")

    if data2:
        main_placeholder.text("📚 Splitting text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(data2)

        main_placeholder.text("🔍 Creating embeddings and vector DB...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        main_placeholder.success("✅ Done! Ask your question below.")

query = main_placeholder.text_input("💬 Ask a question based on the articles:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        st.header("🧠 Answer")
        st.write(result.get("answer", "No answer returned."))

        sources = result.get("sources", "")
        if sources:
            st.subheader("🔗 Sources")
            for source in sources.split("\n"):
                if source.strip():
                    st.write(f"- {source.strip()}")
