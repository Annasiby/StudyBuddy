from __future__ import annotations

"""
StudyBuddy - chat with your lecture notes (PDF/TXT/MD) from the terminal.

RAG pipeline in 5 steps:
  1. LOAD    read each page of your files
  2. CHUNK   cut pages into small overlapping pieces
  3. EMBED   turn every piece into a list of numbers (its "meaning fingerprint")
  4. SEARCH  turn your question into numbers too, find the closest pieces
  5. ASK     give those pieces to Claude and let it answer with citations
"""

import json
import os
import sys
from pathlib import Path

import anthropic
import numpy as np
from pypdf import PdfReader
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# ----------------------------- settings ------------------------------------
DATA_DIR = Path(".studybuddy")          # where we remember your notes
CHUNKS_FILE = DATA_DIR / "chunks.json"  # the text pieces + where they came from
VECTORS_FILE = DATA_DIR / "vectors.npy" # the fingerprints of those pieces
CHUNK_WORDS = 180                       # size of each piece
OVERLAP_WORDS = 40                      # pieces share a little text so ideas aren't cut
TOP_K = 5                               # how many pieces we hand to Claude
MIN_SCORE = 0.25                        # ignore pieces that barely match
MODEL = os.getenv("STUDYBUDDY_MODEL", "claude-sonnet-5-5")
EMBED_MODEL = "all-MiniLM-L6-v2"        # small, free, runs on your laptop

SYSTEM_PROMPT = """You are StudyBuddy, a patient tutor helping a college student \
understand their own lecture notes.

Rules:
- Answer ONLY using the NOTES provided in the user's message.
- After each fact, cite its source like (file.pdf, p.3).
- If the notes don't contain the answer, say so plainly and suggest what to look for. \
Never invent information.
- Explain clearly and simply. Use short paragraphs or bullets. Give a tiny example when it helps."""

console = Console()
_embedder = None


# ----------------------------- embeddings ----------------------------------
def embed(texts: list[str]) -> np.ndarray:
    """Turn text into normalized vectors. Model loads once, on first use."""
    global _embedder
    if _embedder is None:
        with console.status("[cyan]Waking up the meaning-finder (first time is slow)..."):
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ----------------------------- storage -------------------------------------
def load_index() -> tuple[list[dict], np.ndarray | None]:
    if CHUNKS_FILE.exists() and VECTORS_FILE.exists():
        chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        return chunks, np.load(VECTORS_FILE)
    return [], None


def save_index(chunks: list[dict], vectors: np.ndarray | None) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    CHUNKS_FILE.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    if vectors is not None:
        np.save(VECTORS_FILE, vectors)


# ----------------------------- 1. load + 2. chunk --------------------------
def read_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for a file."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


def chunk_text(text: str) -> list[str]:
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, max(len(words), 1), step)
            if words[i:i + CHUNK_WORDS]]


def add_file(path: Path, chunks: list[dict], vectors: np.ndarray | None):
    """Index one file. Re-adding a file replaces its old version."""
    # Drop any older copy of this file
    if vectors is not None and chunks:
        keep = [i for i, c in enumerate(chunks) if c["source"] != path.name]
        chunks = [chunks[i] for i in keep]
        vectors = vectors[keep] if keep else None

    new_chunks = []
    for page_no, text in read_pages(path):
        for piece in chunk_text(text):
            new_chunks.append({"source": path.name, "page": page_no, "text": piece})

    if not new_chunks:
        console.print(f"[yellow]No readable text in {path.name} "
                      "(scanned images? try running OCR first).[/yellow]")
        return chunks, vectors

    with console.status(f"[cyan]Reading {path.name} ({len(new_chunks)} pieces)..."):
        new_vecs = embed([c["text"] for c in new_chunks])

    chunks = chunks + new_chunks
    vectors = new_vecs if vectors is None else np.vstack([vectors, new_vecs])
    console.print(f"[green]✓ Added {path.name}[/green] — {len(new_chunks)} pieces")
    return chunks, vectors


def add_path(target: str, chunks, vectors):
    p = Path(target.strip().strip('"').strip("'")).expanduser()
    if not p.exists():
        console.print(f"[red]Can't find: {p}[/red]")
        return chunks, vectors
    files = [p] if p.is_file() else sorted(
        f for f in p.rglob("*") if f.suffix.lower() in {".pdf", ".txt", ".md"})
    if not files:
        console.print("[yellow]No PDF/TXT/MD files found there.[/yellow]")
    for f in files:
        chunks, vectors = add_file(f, chunks, vectors)
    save_index(chunks, vectors)
    return chunks, vectors


# ----------------------------- 4. search -----------------------------------
def search(question: str, chunks, vectors) -> list[dict]:
    q = embed([question])[0]
    scores = vectors @ q  # vectors are normalized, so this is cosine similarity
    best = np.argsort(scores)[::-1][:TOP_K]
    return [{**chunks[i], "score": float(scores[i])} for i in best if scores[i] >= MIN_SCORE]


# ----------------------------- 5. ask Claude -------------------------------
def ask_claude(client, question: str, hits: list[dict], history: list[dict]) -> str:
    notes = "\n\n".join(
        f"[{h['source']}, p.{h['page']}]\n{h['text']}" for h in hits
    ) or "(no matching notes found)"
    user_msg = f"NOTES:\n{notes}\n\nQUESTION: {question}"

    reply = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=history[-6:] + [{"role": "user", "content": user_msg}],
    ) as stream:
        for piece in stream.text_stream:
            console.print(piece, end="", highlight=False)
            reply += piece
    console.print()
    return reply


# ----------------------------- terminal UI ---------------------------------
HELP = """[bold]Commands[/bold]
  /add <file or folder>   add notes (PDF, TXT, MD)
  /list                   show what's in your library
  /clear                  forget everything
  /help                   show this
  /quit                   leave
Anything else is a question about your notes."""


def show_library(chunks):
    if not chunks:
        console.print("[yellow]Library is empty. Use /add <file or folder>.[/yellow]")
        return
    table = Table(title="Your library")
    table.add_column("File")
    table.add_column("Pieces", justify="right")
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    for name, n in counts.items():
        table.add_row(name, str(n))
    console.print(table)


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[red]Set your key first:[/red]  export ANTHROPIC_API_KEY='sk-ant-...'\n"
                      "(Windows PowerShell:  $env:ANTHROPIC_API_KEY='sk-ant-...')")
        sys.exit(1)

    client = anthropic.Anthropic()
    chunks, vectors = load_index()
    history: list[dict] = []

    console.print(Panel.fit("[bold cyan] StudyBuddy[/bold cyan]\n"
                            "Ask questions about your own lecture notes.\n"
                            "Type /help for commands.", border_style="cyan"))

    for arg in sys.argv[1:]:  # python studybuddy.py notes/ lecture1.pdf
        chunks, vectors = add_path(arg, chunks, vectors)
    show_library(chunks)

    while True:
        try:
            text = Prompt.ask("\n[bold green]You[/bold green]").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not text:
            continue

        if text in {"/quit", "/exit"}:
            break
        if text == "/help":
            console.print(HELP)
        elif text == "/list":
            show_library(chunks)
        elif text == "/clear":
            chunks, vectors, history = [], None, []
            for f in (CHUNKS_FILE, VECTORS_FILE):
                f.unlink(missing_ok=True)
            console.print("[green]Library cleared.[/green]")
        elif text.startswith("/add"):
            arg = text[4:].strip()
            if arg:
                chunks, vectors = add_path(arg, chunks, vectors)
            else:
                console.print("Usage: /add path/to/file.pdf")
        elif text.startswith("/"):
            console.print("[yellow]Unknown command. Try /help[/yellow]")
        else:
            if not chunks:
                console.print("[yellow]Add some notes first: /add my_notes.pdf[/yellow]")
                continue
            hits = search(text, chunks, vectors)
            console.print("\n[bold cyan]StudyBuddy[/bold cyan]")
            try:
                answer = ask_claude(client, text, hits, history)
            except anthropic.APIError as e:
                console.print(f"[red]Claude had a problem: {e}[/red]")
                continue
            history += [{"role": "user", "content": text},
                        {"role": "assistant", "content": answer}]
            if hits:
                srcs = ", ".join(f"{h['source']} p.{h['page']}" for h in hits)
                console.print(f"[dim]📎 Looked at: {srcs}[/dim]")

    console.print("\n[cyan]Good luck studying! [/cyan]")


if __name__ == "__main__":
    main()