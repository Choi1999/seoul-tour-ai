from pathlib import Path
from tqdm import tqdm

# from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader # pip install "unstructured[pdf]"
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import os
os.environ["HF_TOKEN"] = "" # https://share.gemini.google/Gh34Hyl6F6W9

BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)

''' # DirectoryLoader https://share.google/aimode/lcByqRQMwspDzgtzY
loader = DirectoryLoader(
    path="./data",
    glob="**/*.*",
    loader_cls=UnstructuredFileLoader, # pip install unstructured
    loader_kwargs={'languages': ["kor", "eng"]}, # https://share.gemini.google/kA7BelztavMi
    use_multithreading=True,  # Spawns parallel worker threads
    max_concurrency=4         # Adjust based on your CPU cores
)

documents = loader.load()
'''

# https://share.google/aimode/X5EG0DwPY4TJrO61x
loader_mapping = { # https://share.gemini.google/2phhGllF1cFP
    ".pdf": (PyPDFLoader, {}),
    ".csv": (CSVLoader, {'encoding': "utf-8"})
}

# 문서 로드
documents = []

for ext, (loader_cls, loader_kwargs) in loader_mapping.items():
    loader = DirectoryLoader(
        path="./data",
        glob=f"**/*{ext}",
        loader_cls=loader_cls,
        loader_kwargs=loader_kwargs,
        show_progress=True,
        use_multithreading=True # Optional: Speeds up heavy tasks
    )
    documents.extend(loader.load())

print(f"Successfully loaded {len(documents)} documents.")

# 문서 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150
)

docs_splited = splitter.split_documents(documents)
print("# of chunks:", len(docs_splited))

# 임베딩
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
)

DB_PATH = BASE_DIR / "chroma_db"

'''
db = Chroma.from_documents(
    documents = docs_splited,
    embedding = embedding,
    persist_directory = str(DB_PATH)
)
'''

db = Chroma(
    embedding_function=embedding,
    persist_directory=str(DB_PATH)
)

# https://share.gemini.google/ZBestRkquiKk
BATCH_SIZE = 5000 # Chroma가 안정적으로 처리할 수 있는 단위

for i in tqdm(range(0, len(docs_splited), BATCH_SIZE), desc="Embedding & Inserting to Chroma"):
    batch_docs = docs_splited[i : i + BATCH_SIZE]
    db.add_documents(batch_docs)