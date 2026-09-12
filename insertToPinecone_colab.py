# ==============================================================================
# 🚀 AppleSupport Chat Ingestion Pipeline for Google Colab & Pinecone Vector DB
# ==============================================================================
# Instructions for Google Colab:
# 1. Upload this file and your dataset ('twcs.csv') to Google Colab.
# 2. Run: !pip install pinecone-client pandas numpy python-dotenv
# 3. Set your PINECONE_API_KEY environment variable or Colab Secret.
# 4. Run: python insertToPinecone_colab.py --file twcs.csv --upsert
# ==============================================================================

import os
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Try importing Pinecone
try:
    from pinecone import Pinecone
except ImportError:
    print("⚠️ 'pinecone-client' not installed. In Google Colab run: !pip install pinecone-client pandas numpy python-dotenv")

load_dotenv()


def load_tweets_from_csv(csv_path: str) -> pd.DataFrame:
    """Reads CSV archive into a pandas DataFrame using string dtype for IDs."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV file not found at path: '{csv_path}'. "
            "In Google Colab, make sure you uploaded 'twcs.csv' or mounted Google Drive."
        )

    print(f"📂 [Ingestion] Reading CSV dataset: {csv_path}...")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    print(f"✅ Loaded {len(df):,} raw tweets.")
    return df


def filter_applesupport_tweets(
    df: pd.DataFrame, target_brand: str = "AppleSupport"
) -> pd.DataFrame:
    """Vectorized filtering for tweets authored by target_brand or mentioning @target_brand."""
    brand_lower = target_brand.lower()

    author_series = df["author_id"].str.lower()
    text_series = df["text"].str.lower()

    is_author = author_series == brand_lower
    is_mentioned = text_series.str.contains(f"@{brand_lower}", regex=False, na=False)

    mask = np.logical_or(is_author, is_mentioned)
    filtered_df = df[mask].copy()

    filtered_df["is_brand_author"] = is_author[mask]
    filtered_df["is_inbound"] = filtered_df["inbound"].str.lower() == "true"

    print(f"✅ Isolated {len(filtered_df):,} '{target_brand}' related tweets.")
    return filtered_df


def reconstruct_user_brand_pairs(
    df: pd.DataFrame, target_brand: str = "AppleSupport"
) -> List[Dict[str, Any]]:
    """
    Extracts 1 resolution record per conversation thread, starting from the
    VERY FIRST message from the user up to the VERY FIRST reply from the brand.

    - '_id': tweet_id of the initial starting user message.
    - 'text': combined user/customer messages up to the first brand reply.
    - 'brand_reply': text of the very first reply from the brand (stored in metadata).
    """
    brand_lower = target_brand.lower()

    # Build fast O(1) dictionary mapping: tweet_id -> row_dict
    tweet_map = {row["tweet_id"]: row for row in df.to_dict("records")}

    # Find root user tweets (inbound customer tweets mentioning brand with no parent in dataset)
    root_user_tweets = []
    for tid, t in tweet_map.items():
        if t["author_id"].lower() != brand_lower and (f"@{brand_lower}" in t["text"].lower()):
            parent_id = t.get("in_response_to_tweet_id", "").strip()
            if not parent_id or parent_id not in tweet_map:
                root_user_tweets.append(t)

    records = []
    for root in root_user_tweets:
        user_messages = [root]
        first_brand_reply = None

        queue = [root["tweet_id"]]
        visited = set()

        while queue and not first_brand_reply:
            curr_id = queue.pop(0)
            if curr_id in visited or curr_id not in tweet_map:
                continue
            visited.add(curr_id)
            t = tweet_map[curr_id]

            if t["author_id"].lower() == brand_lower:
                first_brand_reply = t
                break
            elif curr_id != root["tweet_id"]:
                user_messages.append(t)

            resp_ids_str = t.get("response_tweet_id", "").strip()
            if resp_ids_str:
                for r_id in resp_ids_str.split(","):
                    r_id = r_id.strip()
                    if r_id in tweet_map and r_id not in visited:
                        queue.append(r_id)

        if first_brand_reply:
            user_text_lines = [
                f"Customer ({t['author_id']}): {t['text']}" for t in user_messages
            ]
            user_text_combined = "\n".join(user_text_lines)

            record = {
                "_id": root["tweet_id"],  # ID of initial starting user message
                "text": user_text_combined,  # Combined user messages up to first brand reply
                "brand_reply": first_brand_reply["text"],  # Very first reply from the brand
                "brand": target_brand,
                "reply_tweet_id": first_brand_reply["tweet_id"],
                "created_at": root.get("created_at", ""),
                "user_tweet_count": len(user_messages),
            }
            records.append(record)

    print(f"✅ Extracted {len(records):,} clean first-user to first-brand conversation records.")
    return records


def upsert_records_to_pinecone(
    records: List[Dict[str, Any]],
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
    batch_size: int = 100,
) -> int:
    """
    Batch-upserts reconstructed conversation records to Pinecone vector DB
    using Pinecone's Integrated Hosted Inference.
    """
    if not records:
        print("⚠️ No records to upsert.")
        return 0

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("PINECONE_API_KEY")
        except Exception:
            pass

    if not api_key:
        import getpass
        api_key = getpass.getpass("Enter your Pinecone API Key: ")

    if not index_name:
        index_name = os.getenv("PINECONE_INDEX_NAME", "customer-support-resolutions")
    if not namespace:
        namespace = os.getenv("PINECONE_NAMESPACE", "__default__")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # Pinecone Integrated Hosted Inference (upsert_records) requires batch_size <= 96
    batch_size = min(batch_size, 96)

    total_records = len(records)
    total_batches = (total_records + batch_size - 1) // batch_size
    print(f"\n🌲 Connecting to Pinecone Index '{index_name}' (Namespace: '{namespace}')...")
    print(f"🚀 Upserting {total_records:,} records in {total_batches:,} batches (batch_size={batch_size})...")

    upserted_count = 0
    for i in range(0, total_records, batch_size):
        batch = records[i : i + batch_size]
        index.upsert_records(namespace=namespace, records=batch)
        upserted_count += len(batch)
        batch_num = i // batch_size + 1
        print(f"  └─ Batch {batch_num}/{total_batches}: Upserted {upserted_count:,}/{total_records:,} records.")

    print(f"✅ Successfully inserted {upserted_count:,} records into Pinecone DB index '{index_name}'.")
    return upserted_count


def process_colab_pipeline(
    csv_path: str,
    brand: str = "AppleSupport",
    limit: Optional[int] = None,
    upsert: bool = False,
    upsert_limit: Optional[int] = None,
    batch_size: int = 100,
    namespace: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Runs the full Google Colab ingestion, reconstruction, and upsert pipeline."""
    df = load_tweets_from_csv(csv_path)
    filtered_df = filter_applesupport_tweets(df, target_brand=brand)
    records = reconstruct_user_brand_pairs(df, target_brand=brand)

    if limit and limit > 0:
        records = records[:limit]
        print(f"ℹ️ Applied limit: processing top {len(records):,} records.")

    print("\n--- 📋 Preview of Reconstructed Records (Showing first 2) ---")
    for i, record in enumerate(records[:2]):
        print(f"\n==================== [Record #{i+1}] ====================")
        print(f"ID (Initial User Tweet ID): {record['_id']}")
        print(f"Brand: {record['brand']} | First Reply Tweet ID: {record['reply_tweet_id']}")
        print(f"User Tweets Count: {record['user_tweet_count']}")
        print("\n🔹 TEXT FIELD (User Messages up to First Brand Reply):")
        print(record["text"])
        print("\n🔹 METADATA 'brand_reply' (First Brand Reply):")
        print(record["brand_reply"])

    if upsert and records:
        records_to_upsert = records
        if upsert_limit and upsert_limit > 0:
            records_to_upsert = records[:upsert_limit]
            print(f"\nℹ️ Limiting Pinecone upsert to {len(records_to_upsert)} records.")

        upsert_records_to_pinecone(
            records_to_upsert,
            namespace=namespace,
            batch_size=batch_size,
        )
    elif not upsert:
        print("\n💡 Run complete. Pass `--upsert` flag to insert records into Pinecone Vector DB.")

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Google Colab Pipeline: Ingest, filter, reconstruct AppleSupport chats and insert to Pinecone DB."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="twcs.csv",
        help="Path to archive CSV file (default: twcs.csv)",
    )
    parser.add_argument(
        "--brand",
        type=str,
        default="AppleSupport",
        help="Brand name (default: AppleSupport)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on maximum records to process",
    )
    parser.add_argument(
        "--upsert",
        action="store_true",
        help="Perform insertion into Pinecone Vector DB",
    )
    parser.add_argument(
        "--upsert-limit",
        type=int,
        default=None,
        help="Limit number of records to upsert to Pinecone (e.g. --upsert-limit 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for Pinecone upsert (default: 100)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Pinecone namespace (default: from PINECONE_NAMESPACE env or __default__)",
    )

    args = parser.parse_args()

    process_colab_pipeline(
        csv_path=args.file,
        brand=args.brand,
        limit=args.limit,
        upsert=args.upsert,
        upsert_limit=args.upsert_limit,
        batch_size=args.batch_size,
        namespace=args.namespace,
    )
