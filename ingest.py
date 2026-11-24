from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

#load pdf
pdf_path = "data/sample.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("pages", len(documents))

#split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=150,
    add_start_index=True
)

chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))

#initialize embedddings
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

#initialize chroma
vector_store =Chroma(
    collection_name="pdf_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

#store documents
ids = [
    f"{Path(pdf_path).stem}-{i}"
    for i in range(len(chunks))
]

vector_store.add_documents(
    documents=chunks,
    ids=ids
)
print("PDF successfully indexed")