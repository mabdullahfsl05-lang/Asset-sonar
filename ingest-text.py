import os
from dotenv import load_dotenv
load_dotenv()
os.environ["USER_AGENT"] = "AssetSonarMCP/1.0"

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

def ingest_articles_to_pinecone():
    
    try:
        with open("urls.txt", "r") as file:
            urls = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        print("❌ Error: Could not find 'urls.txt'. Please create it in the same folder.")
        return
        
    if not urls:
        print("❌ Error: 'urls.txt' is empty! Please add some links.")
        return

    loader = AsyncHtmlLoader(urls)
    docs = loader.load()
    
    docs = Html2TextTransformer().transform_documents(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    split_docs = text_splitter.split_documents(docs)

    print("🧠 Creating embeddings and connecting to Pinecone...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    index_name = os.environ.get("PINECONE_INDEX_NAME")


    PineconeVectorStore.from_documents(
        embedding=embeddings,
        index_name=index_name,
        documents=split_docs
    )
    
if __name__ == "__main__":
    ingest_articles_to_pinecone()