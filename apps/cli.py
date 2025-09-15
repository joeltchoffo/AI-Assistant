from __future__ import annotations
import glob
import typer
from rich import print
from tqdm import tqdm
from rag.ingest import extract_pdf_chunks, to_dicts
from rag.store_faiss import build_index
from rag.search import search

app = typer.Typer(help="RAG CLI")

@app.command()
def ingest(path: str = typer.Argument("data/*.pdf", help="Glob-Pfad zu PDFs"),
           index_dir: str = typer.Option("indices/faiss", help="Ziel für FAISS-Index")):
    files = glob.glob(path)
    if not files:
        print("[yellow]Keine Dateien gefunden.[/yellow]")
        raise typer.Exit(code=1)

    all_chunks = []
    for fp in files:
        print(f"[cyan]Parse[/cyan] {fp}")
        chunks = extract_pdf_chunks(fp, max_tokens=600, overlap=100)
        all_chunks.extend(to_dicts(chunks))

    print(f"[green]Baue Index[/green] ({len(all_chunks)} Chunks)…")
    build_index(all_chunks, index_dir=index_dir)
    print(f"[green]Fertig.[/green]")

@app.command()
def query(q: str,
          k: int = typer.Option(5, help="Top-k"),
          index_dir: str = typer.Option("indices/faiss", help="FAISS-Verzeichnis")):
    hits = search(q, top_k=k, index_dir=index_dir)
    for i, h in enumerate(hits, 1):
        print(f"\n[i]{i}[/i]  [bold]{h['score']:.3f}[/bold]  {h['doc_id']}  S.{h['page_start']}")
        print(h["text"][:500].replace("\n", " ") + ("…" if len(h["text"])>500 else ""))

if __name__ == "__main__":
    app()
