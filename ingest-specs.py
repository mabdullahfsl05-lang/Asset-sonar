import os
import json
import requests
from typing import Any
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
LOCAL_JSON_PATH = "inventory.json"


def get_site_access() -> dict[str, Any]:
    """Retrieves secure access credentials for AssetSonar."""
    token = os.environ.get("ASSETSONAR_TOKEN")
    subdomain = os.environ.get("ASSETSONAR_SUBDOMAIN")

    if not token or not subdomain:
        raise RuntimeError(
            "Missing ASSETSONAR_TOKEN or ASSETSONAR_SUBDOMAIN environment variable."
        )

    return {
        "headers": {
            "token": token,
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        },
        "base_url": f"https://{subdomain}.ez-oi.com",
    }


def fetch_all_assets() -> list[dict[str, Any]]:
    """Fetches every asset from AssetSonar, paginating 25 at a time."""
    site = get_site_access()
    url = f"{site['base_url']}/assets.api"

    all_items: list[dict[str, Any]] = []
    page = 1

    while True:
        params = {
            "page": page,
            "include_custom_fields": "true",
            "show_hardware_details": "true",
            "include_system_details": "true",
        }
        response = requests.get(url, headers=site["headers"], params=params)
        response.raise_for_status()
        data = response.json()

        assets = data.get("assets", []) if isinstance(data, dict) else data
        if not assets:
            break

        all_items.extend(assets)

        if len(assets) < 25:
            break
        page += 1

    return all_items


def format_asset_to_text(item: dict[str, Any]) -> str:
    """Converts one raw AssetSonar asset record into a natural-language string."""
    
    # Handle whether the API returns a direct dictionary or wraps it in {"asset": {...}}
    asset_data = item.get("asset", item)
    
    asset_id = asset_data.get("sequence_num", asset_data.get("id", "N/A"))
    name = asset_data.get("name", "Unknown Asset")
    
    # Group name sometimes lives outside the main asset object depending on the endpoint
    group = item.get("group_name") or asset_data.get("group_name") or "Uncategorized"

    hardware = asset_data.get("hardware_details") or {}
    system = asset_data.get("system_details") or {}

    # Extract directly from the top level of the asset data payload
    ram = asset_data.get("RAM") or hardware.get("ram") or "N/A"
    gpu = asset_data.get("GPU") or hardware.get("gpu") or "N/A"
    cpu = asset_data.get("CPU") or system.get("cpu") or "N/A"

    return (
        f"Asset ID {asset_id} | Name: {name} | Group: {group} | "
        f"RAM: {ram} | GPU: {gpu} | CPU: {cpu}"
    )

def fetch_and_ingest_inventory():
    """Fetches all assets from AssetSonar, clears old vectors, and ingests the new inventory."""
    if not INDEX_NAME:
        raise RuntimeError("Missing PINECONE_INDEX_NAME environment variable.")

    print("Fetching live inventory from AssetSonar...")
    raw_assets = fetch_all_assets()
    print(f"Successfully retrieved {len(raw_assets)} assets.")

    if not raw_assets:
        print("No assets were fetched. Exiting.")
        return

    formatted = [format_asset_to_text(item) for item in raw_assets]

    with open(LOCAL_JSON_PATH, "w") as f:
        json.dump(formatted, f)
    print(f"Saved local inventory to {LOCAL_JSON_PATH} for BM25 keyword search.")

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("Missing PINECONE_API_KEY environment variable.")

    # 1. Connect to the Index
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(INDEX_NAME)
    
    # 2. Wipe the old vectors completely
    print(f"Clearing old vectors from Pinecone index '{INDEX_NAME}'...")
    try:
        index.delete(delete_all=True)
        print("Old vectors successfully deleted.")
    except Exception as e:
        print(f"Note: Could not delete old vectors (index might be empty). Details: {e}")

    # 3. Embed and upload the fresh data
    print("Embedding and uploading fresh data to Pinecone...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    PineconeVectorStore.from_texts(
        texts=formatted,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    print(f"Successfully ingested {len(formatted)} assets into Pinecone.")


if __name__ == "__main__":
    fetch_and_ingest_inventory()