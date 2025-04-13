import os
import gradio as gr
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

# Dynamically construct the path to the vector DB folder
current_dir = os.path.dirname(os.path.abspath(__file__))
my_vector_db_path = os.path.join(current_dir, "../my_vector_db")

# Load FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = FAISS.load_local(my_vector_db_path, embedding_model, allow_dangerous_deserialization=True)

# Dynamically construct the path to the GGUF model folder
# Note: You can change the model name to any other supported by HuggingFace
local_model_path = os.path.join(current_dir, "../model/phi-2.Q4_K_M.gguf")

# Load GGUF Phi-2 model
phi2llm = LlamaCpp(
    model_path=local_model_path,
    temperature=0.7,
    max_tokens=512,
    n_ctx=2048,
    top_p=0.9,
    verbose=True
)

# Build RAG pipeline
rag_chain = RetrievalQA.from_chain_type(
    llm=phi2llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=False
)

def chat(query):
    result = rag_chain(query)
    return result["result"]

# UI
gr.Interface(fn=chat, inputs="text", outputs="text", title="FAISS Vector Based RAG Chain Application").launch(server_name="0.0.0.0")
