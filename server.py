from fastmcp import FastMCP
from dotenv import load_dotenv
load_dotenv()
import os
import json
import requests
from typing import Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

mcp = FastMCP("AssetSonar")

# Shared, absolute path — independent of whatever directory the server is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_JSON_PATH = os.path.join(BASE_DIR, "inventory.json")

# Loaded once, reused everywhere.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_site_access():
    token = os.environ.get("ASSETSONAR_TOKEN")
    subdomain = os.environ.get("ASSETSONAR_SUBDOMAIN")

    if not token or not subdomain:
        raise RuntimeError(
            "Missing ASSETSONAR_TOKEN or ASSETSONAR_SUBDOMAIN environment variables. "
            "Set them before starting the server."
        )

    return {
        "headers": {
            "token": token,
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        },
        "base_url": f"https://{subdomain}.ez-oi.com",
    }


def _handle_response(response: requests.Response) -> dict[str, Any]:
    """Raise a clear, MCP-friendly error instead of a raw HTTPError."""
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"AssetSonar API error {response.status_code} for {response.url}: "
            f"{response.text[:500]}"
        ) from exc
    return response.json()


@mcp.tool
def create_asset(
    name: str,
    group_id: int,
    purchased_on: str,
    sub_group_id: int | None = None,
    location_id: int | None = None,
    manufacturer: str | None = None,
    bios_serial_number: str | None = None,
    image_url: str | None = None,
    document_url1: str | None = None,
    document_url2: str | None = None,
    identifier: str | None = None,
    custom_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Creates an asset in AssetSonar.

    name: Name of the asset.
    group_id: ID of the Group the asset belongs to.
    purchased_on: Date the asset was purchased, formatted as MM/DD/YYYY.
    sub_group_id: Optional ID of the sub-group.
    location_id: Optional ID of the asset's location.
    manufacturer: Optional manufacturer name.
    bios_serial_number: Optional BIOS serial number.
    image_url: Optional URL of an image to attach to the asset.
    document_url1: Optional URL of a document to attach to the asset.
    document_url2: Optional URL of a second document to attach to the asset.
    identifier: Optional custom identifier for the asset.
    custom_fields: Optional dictionary mapping custom field names to their values.
    """
    site = get_site_access()
    dest_url = f"{site['base_url']}/assets.api"

    data: dict[str, Any] = {
        "fixed_asset[name]": name,
        "fixed_asset[group_id]": group_id,
        "fixed_asset[purchased_on]": purchased_on,
    }

    if sub_group_id is not None:
        data["fixed_asset[sub_group_id]"] = sub_group_id
    if location_id is not None:
        data["fixed_asset[location_id]"] = location_id
    if manufacturer is not None:
        data["fixed_asset[manufacturer]"] = manufacturer
    if bios_serial_number is not None:
        data["fixed_asset[bios_serial_number]"] = bios_serial_number
    if image_url is not None:
        data["fixed_asset[image_url]"] = image_url
    if identifier is not None:
        data["fixed_asset[identifier]"] = identifier

    if custom_fields is not None:
        for key, value in custom_fields.items():
            data[f"cust_attr[{key}]"] = value

    docs = [url for url in (document_url1, document_url2) if url is not None]
    if docs:
        data["fixed_asset[document_urls][]"] = docs

    response = requests.post(dest_url, data=data, headers=site["headers"])
    return _handle_response(response)


@mcp.tool
def retrieve_all_assets(page_num: int = 1) -> dict[str, Any]:
    """Retrieves list of all assets in pages. Only 25 assets per page."""
    site = get_site_access()
    dest_url = f"{site['base_url']}/assets.api"

    params = {
        "page": page_num,
        "include_custom_fields": "true",
        "show_document_urls": "true",
        "show_image_urls": "true",
        "show_document_details": "true",
        "include_system_details": "true",   # Added to fetch CPU/OS details
        "include_hardware_details": "true"  # Added to fetch RAM/GPU/Battery details
    }

    response = requests.get(url=dest_url, params=params, headers=site["headers"])
    return _handle_response(response)


@mcp.tool
def retrieve_checked_out_assets(page_num: int = 1) -> dict[str, Any]:
    """Retrieves list of all assets in pages that are flagged as checked out. Only 25 assets per page."""
    site = get_site_access()
    dest_url = f"{site['base_url']}/assets/filter.api"

    params = {
        "page": page_num,
        "status": "checked_out",
        "include_system_details": "true",   # Added to fetch CPU/OS details
        "include_hardware_details": "true"  # Added to fetch RAM/GPU/Battery details
    }

    response = requests.get(url=dest_url, params=params, headers=site["headers"])
    return _handle_response(response)


@mcp.tool
def retrieve_asset_details(asset_num: int) -> dict[str, Any]:
    """Retrieves the details of a specific asset given the asset number."""
    site = get_site_access()
    dest_url = f"{site['base_url']}/assets/{asset_num}.api"

    params = {
        "show_document_urls": "true",
        "show_image_urls": "true",
        "show_document_details": "true",
        "include_custom_fields": "true",
        "include_system_details": "true",
        "include_hardware_details": "true",
    }

    response = requests.get(url=dest_url, params=params, headers=site["headers"])
    return _handle_response(response)


@mcp.tool
def update_asset_gps_coordinates(
    asset_num: int,
    latitude: float,
    longitude: float,
    gps_asset_id: int | None = None,
) -> dict[str, Any]:
    """Updates the longitude and latitude coordinates of an asset using an asset number.

    asset_num: ID of the asset.
    latitude: New latitude value.
    longitude: New longitude value.
    gps_asset_id: Optional GPS asset ID.
    """
    site = get_site_access()
    dest_url = f"{site['base_url']}/assets/{asset_num}/gps_coordinates.api"

    data: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
    }

    if gps_asset_id is not None:
        data["gps_asset_id"] = gps_asset_id

    response = requests.patch(url=dest_url, data=data, headers=site["headers"])
    return _handle_response(response)


def get_pinecone_retriever():
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not index_name:
        print("Warning: PINECONE_INDEX_NAME is not set. KB queries will fail.")
        return None

    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


kb_retriever = get_pinecone_retriever()


@mcp.tool
def query_knowledge_base(search_query: str) -> str:
    """Queries the internal IT knowledge base for policies, guides, and articles.
    search_query: The specific question or keywords to search for.
    Returns the relevant article excerpts along with their direct source links.
    """
    if not kb_retriever:
        return "Error: Database not configured properly."

    retrieved_docs = kb_retriever.invoke(search_query)

    if not retrieved_docs:
        return "No relevant articles found in the knowledge base."

    formatted_results = []
    for i, doc in enumerate(retrieved_docs, 1):
        content = doc.page_content
        link = doc.metadata.get('url', 'No link available')

        result_text = f"Result {i}:\nExcerpt: {content}\nSource Link: {link}"
        formatted_results.append(result_text)

    return "\n\n---\n\n".join(formatted_results)


def setup_rag_pipeline():
    """Builds the Two-Stage Hybrid + Reranker Pipeline."""

    if not os.path.exists(LOCAL_JSON_PATH):
        raise FileNotFoundError(f"{LOCAL_JSON_PATH} missing. Please run ingest-specs.py first.")

    with open(LOCAL_JSON_PATH, "r") as f:
        inventory_texts = json.load(f)

    docs = [Document(page_content=text) for text in inventory_texts]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 15

    index_name = os.environ.get("PINECONE_INDEX_NAME")
    if not index_name:
        raise RuntimeError("Missing PINECONE_INDEX_NAME environment variable.")

    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if not cohere_api_key:
        raise RuntimeError("Missing COHERE_API_KEY environment variable.")

    pinecone_vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    pinecone_retriever = pinecone_vectorstore.as_retriever(search_kwargs={"k": 15})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, pinecone_retriever],
        weights=[0.4, 0.6],
    )

    compressor = CohereRerank(
        cohere_api_key=cohere_api_key,
        model="rerank-v3.5",
        top_n=3,
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )


# Build the pipeline once at server boot, but don't let a failure here take down
# the whole MCP server — other tools should keep working regardless.
try:
    retriever_pipeline = setup_rag_pipeline()
except Exception as e:
    print(f"Warning: RAG pipeline failed to initialize: {e}")
    retriever_pipeline = None


@mcp.tool()
def recommend_hardware(query: str) -> str:
    """
    Searches the live IT AssetSonar inventory for the best hardware match based on user requirements.
    Provide a detailed query string representing the user's needs (e.g. "I need an 8gb ram with gpu for coding").
    """
    if retriever_pipeline is None:
        return "Error: hardware recommendation pipeline is not configured. Check server logs."

    results = retriever_pipeline.invoke(query)

    if not results:
        return "No matching hardware found in the current inventory."

    context = "--- TOP RECOMMENDED HARDWARE ---\n"
    for i, doc in enumerate(results):
        context += f"{i+1}. {doc.page_content}\n"

    return context

@mcp.tool
def update_asset(
    asset_num: int,
    name: str | None = None,
    group_id: int | None = None,
    sub_group_id: int | None = None,
    purchased_on: str | None = None,
    location_id: int | None = None,
    manufacturer: str | None = None,
    bios_serial_number: str | None = None,
    image_url: str | None = None,
    document_url1: str | None = None,
    document_url2: str | None = None,
    custom_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Updates an existing asset in AssetSonar.

    asset_num: ID of the asset to update.
    name: Optional new name of the asset.
    group_id: Optional ID of the Group.
    sub_group_id: Optional ID of the sub-group.
    purchased_on: Optional date the asset was purchased, formatted as MM/DD/YYYY.
    location_id: Optional ID of the asset's location.
    manufacturer: Optional manufacturer name.
    bios_serial_number: Optional BIOS serial number.
    image_url: Optional URL of an image to attach to the asset.
    document_url1: Optional URL of a document to attach.
    document_url2: Optional URL of a second document to attach.
    custom_fields: Optional dictionary mapping custom field names to their values (e.g., {"RAM": "16GB", "CPU": "Intel i7"}).
    """
    site = get_site_access()
    dest_url = f"{site['base_url']}/assets/{asset_num}.api"

    data: dict[str, Any] = {}

    if name is not None:
        data["fixed_asset[name]"] = name
    if group_id is not None:
        data["fixed_asset[group_id]"] = group_id
    if sub_group_id is not None:
        data["fixed_asset[sub_group_id]"] = sub_group_id
    if purchased_on is not None:
        data["fixed_asset[purchased_on]"] = purchased_on
    if location_id is not None:
        data["fixed_asset[location_id]"] = location_id
    if manufacturer is not None:
        data["fixed_asset[manufacturer]"] = manufacturer
    if bios_serial_number is not None:
        data["fixed_asset[bios_serial_number]"] = bios_serial_number
    if image_url is not None:
        data["fixed_asset[image_url]"] = image_url

    # Iterates over the dictionary and maps the values to the API's custom attribute format
    if custom_fields is not None:
        for key, value in custom_fields.items():
            data[f"cust_attr[{key}]"] = value

    docs = [url for url in (document_url1, document_url2) if url is not None]
    if docs:
        data["fixed_asset[document_urls][]"] = docs

    if not data:
        return {"message": "No update fields provided."}

    response = requests.put(url=dest_url, data=data, headers=site["headers"])
    return _handle_response(response)


if __name__ == "__main__":
    mcp.run()