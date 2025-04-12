import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Dynamically construct the path to the docs folder
current_dir = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.join(current_dir, "../docs")

# Load all .txt/.md/.html files from the docs folder
loader = DirectoryLoader(docs_path, glob="**/*.*", loader_cls=UnstructuredFileLoader)
documents = loader.load()

print(documents.__len__(), "documents loaded")

# Split documents into chunks of text for vectorization using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(chunks.__len__(), "chunks created")

# Vectorize the chunks using HuggingFaceEmbeddings Model (all-MiniLM-L6-v2)
# Note: You can change the model name to any other supported by HuggingFace
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Vectorizing chunks...")

# Create a FAISS vector store from the chunks and embeddings
vector_db = FAISS.from_documents(chunks, embeddings)

print("Vectorization complete!")

# Save the vector store locally
# Note: You can change the path to any other directory where you want to save the vector store
my_vector_db_path = os.path.join(current_dir, "../my_vector_db")
vector_db.save_local(my_vector_db_path)

print("Vector DB created successfully!")
