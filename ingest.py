import argparse
import glob
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from embeddings import embed_documents

def get_index():
    if not config.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is missing - set it in .env.")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if config.PINECONE_INDEX not in [i.name for i in pc.list_indexes()]:
        print(f"Creating Pinecone index '{config.PINECONE_INDEX}' "
              f"(dim={config.EMBEDDING_DIM}, metric={config.PINECONE_METRIC}, "
              f"{config.PINECONE_CLOUD}/{config.PINECONE_REGION})...")
        pc.create_index(
            name=config.PINECONE_INDEX,
            dimension=config.EMBEDDING_DIM,
            metric=config.PINECONE_METRIC,
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
    idx = pc.Index(config.PINECONE_INDEX)
    if idx.describe_index_stats().get("dimension") not in (None, config.EMBEDDING_DIM):
        raise RuntimeError("Pinecone index dimension != EMBEDDING_DIM - delete it or fix .env.")
    return idx

def ingest(pdf_path=None) -> dict:
    pdf_path = Path(pdf_path or config.PDF_PATH)
    if pdf_path.is_dir():
        pdfs = sorted(glob.glob(str(pdf_path / "*.pdf")))
    elif "*" in str(pdf_path):
        pdfs = sorted(glob.glob(str(pdf_path)))
    else:
        pdfs = [str(pdf_path)]
    if not pdfs or not Path(pdfs[0]).exists():
        raise FileNotFoundError(f"No PDF at {pdf_path}. Download the eBook into data/ first.")

    docs = []
    for p in pdfs:
        docs.extend(PyPDFLoader(p).load())
    print(f"Loaded {len(docs)} pages from {pdfs}")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents(docs)
    print(f"Split into {len(chunks)} chunks — embedding with {config.EMBEDDING_MODEL}...")

    index = get_index()

    existing = set()
    try:
        for page in index.list():
            for v in page.vectors or []:
                existing.add(v.id if hasattr(v, "id") else v.get("id"))
    except Exception as e:
        print(f"(could not list existing vectors: {e} — ingesting all)")
    todo = [(i, c) for i, c in enumerate(chunks) if f"chunk-{i}" not in existing]
    print(f"{len(existing)} already in index, {len(todo)} to embed/upsert.")

    done = 0
    for j in range(0, len(todo), 10):
        batch = todo[j:j + 10]
        vecs = embed_documents([c.page_content for _, c in batch])
        records = [(
            f"chunk-{i}",
            vec,
            {"text": c.page_content,
             "page": c.metadata.get("page", "?"),
             "source": Path(c.metadata.get("source", "")).name,
             "chunk_id": i},
        ) for (i, c), vec in zip(batch, vecs)]
        index.upsert(records)
        done += len(records)
        print(f"Upserted {done}/{len(todo)}...")
    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", len(existing) + done)
    print(f"Done — index '{config.PINECONE_INDEX}' now holds {total} vectors: {stats}")
    return {"pages": len(docs), "chunks": total, "index": config.PINECONE_INDEX}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(config.PDF_PATH))
    print(ingest(ap.parse_args().pdf))
