import argparse
import os
import shutil
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader)
from langchain_text_splitters import TokenTextSplitter
from langchain.schema.document import Document
from embedding_function import get_embedding_function
from langchain_chroma import Chroma



DATA_PATH ="data"
CHROMA_PATH = "chroma"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",help="Reset the database")
    args = parser.parse_args()
    if args.reset:
        print("Clearing database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    print("Ingestion complete.")


def load_documents():
    documents = []

    # Load PDF files
    pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents.extend(pdf_loader.load())

    # Load TXT files
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".txt"):
            txt_loader = TextLoader(os.path.join(DATA_PATH, filename), encoding="utf-8")
            documents.extend(txt_loader.load())

    # Load DOCX files
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".docx"):
            docx_loader = UnstructuredWordDocumentLoader(os.path.join(DATA_PATH, filename))
            documents.extend(docx_loader.load())

    # Load Markdown files
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".md"):
            md_loader = UnstructuredMarkdownLoader(os.path.join(DATA_PATH, filename))
            documents.extend(md_loader.load())

    return documents

def split_documents(documents: list[Document]):
    text_splitter = TokenTextSplitter(
        chunk_size=1024,       # tokens per chunk
        chunk_overlap=128,     # token overlap
    )
    return text_splitter.split_documents(documents)



def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory = CHROMA_PATH,embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks=[]
    for chunk in  chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
    
    if len(new_chunks):
        print(f"Adding new documents:{len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for  chunk in new_chunks]
        db.add_documents(new_chunks,ids=new_chunk_ids)
    else:
        print("No new documents to add.")  



def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index=0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index +=1
        else:
            current_chunk_index=0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    return chunks    



def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)



if __name__ == "__main__":
    main()