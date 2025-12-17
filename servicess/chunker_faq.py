import os
import re
import json
import hashlib
import argparse
from pathlib import Path

# ---------- Helpers ----------

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def approx_token_count(text: str) -> int:
    # Simple approximation: words * 1.3 (tokens are usually subwords)
    # This avoids needing the heavy tiktoken library just for this script
    words = re.findall(r"\S+", text)
    return max(1, int(len(words) * 1.3))

# ---------- FAQ Parser ----------

def parse_faq_text(file_path: str) -> list:
    """
    Reads a text file and extracts Q&A pairs.
    Assumes format:
    Q: Question text?
    A: Answer text...
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Normalize newlines
    content = content.replace('\r\n', '\n')
    
    # Split by "Q:" or "Question:" at the start of a line
    # The regex looks for a newline followed by Q: or Question:
    raw_blocks = re.split(r'\n(?=Q:|Question:)', content)
    
    chunks = []
    chunk_counter = 0
    
    for block in raw_blocks:
        block = block.strip()
        if not block: continue
        
        # Check if it actually looks like a Question
        if not (block.startswith("Q:") or block.startswith("Question:")):
            continue

        # Extract Question and Answer parts
        # We split at the first "A:" or "Answer:"
        parts = re.split(r'\n(?=A:|Answer:)', block, maxsplit=1)
        
        if len(parts) == 2:
            question = parts[0].strip()
            answer = parts[1].strip()
            
            # Combine them for the 'text' field (used for embedding)
            full_text = f"{question}\n{answer}"
            
            # Create the chunk object strictly matching your schema
            chunk = {
                "doc_id": "FAQ",
                "chunk_id": f"FAQ_c{chunk_counter:04d}",
                "title": question, # The Question acts as the Title
                "subheading": None,
                "start_offset": 0,
                "end_offset": len(full_text),
                "tokens": approx_token_count(full_text),
                "section_tags": ["faq", "actionable", "policy"], # Auto-tagging
                "effective_date": None,
                "jurisdiction": "India",
                "sensitive": False,
                "source_citation": "FAQ",
                "checksum": sha256_text(full_text),
                "text": full_text
            }
            
            chunks.append(chunk)
            chunk_counter += 1
            
    return chunks

# ---------- Runner ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="input_pdfs/FAQ.txt", help="Path to FAQ.txt")
    parser.add_argument("--output-dir", default="chunks_out", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "faq_chunks.jsonl"

    if not input_path.exists():
        print(f"[!] Error: File not found at {input_path}")
        return

    print(f"[+] Reading FAQ from: {input_path}")
    chunks = parse_faq_text(str(input_path))
    
    if chunks:
        with open(output_file, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        print(f"[+] Success! Generated {len(chunks)} rich FAQ chunks.")
        print(f"[+] Saved to: {output_file}")
        print("\n--> NEXT STEP: You can now upsert this file specifically.")
    else:
        print("[!] Warning: No 'Q: ... A: ...' patterns found. Check your text file formatting.")

if __name__ == "__main__":
    main()