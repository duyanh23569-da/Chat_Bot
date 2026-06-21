import pdfplumber
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DATA_FOLDER = "data/"
VECTOR_FOLDER = "vectorstore/"

def load_documents(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.endswith(".pdf"):

            content = ""

            with pdfplumber.open(filepath) as pdf:

                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
                    tables = page.extract_tables()

                    for table in tables:
                        for row in table:
                            if row:
                                row_text = " | ".join(
                                    str(cell) if cell else ""
                                    for cell in row
                                )
                                content += row_text + "\n"

            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": filename}
                )
            )

            print(f"📄 Đã đọc: {filename}")

    return documents

def build_vectorstore():
    docs = load_documents(DATA_FOLDER)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )# chia thành các đoạn nhỏ nếu chunk quá nhỏ thì mất nghĩa, quá to thì search sai
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    ) # không có thằng này thì chỉ tìm theo key;

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )# thằng chroma sẽ lưu các vector số phía trên; nó sẽ tìm đoạn có gần nghĩa nhất

    print("✅ Đã tạo vector database!")


if __name__ == "__main__":
    build_vectorstore()