"""
chunker.py
Full chunker implementing the strategy described earlier.

Requirements:
  pip install pymupdf tiktoken

Usage:
  python chunker.py \
    --input-dir /mnt/data \
    --output ./chunks_out \
    --min-tokens 500 --max-tokens 800 --overlap 100

Notes:
  - For accurate token counts use tiktoken. If tiktoken isn't available,
    the script approximates token counts by whitespace-splitting.
  - Special enrichment is applied when filename contains 'Cancellation' (case-insensitive).
"""

import os
import re
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any

# Try to import optional libs
try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError("PyMuPDF (fitz) is required. pip install pymupdf") from e

try:
    import tiktoken
    TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    TOKEN_ENCODER = None

# ---------- Helpers ----------

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def approx_token_count(text: str) -> int:
    if TOKEN_ENCODER:
        return len(TOKEN_ENCODER.encode(text))
    # fallback approximation: 0.75 * number of whitespace tokens (heuristic)
    words = re.findall(r"\S+", text)
    return max(1, int(len(words) * 0.75))

def sentence_split(text: str) -> List[str]:
    # Simple sentence splitter: split after .!? but keep abbreviations naive approach
    # Works well enough for chunking purposes.
    parts = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z0-9\"\'\(\[])', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def smart_truncate(tokens: List[str], max_tokens:int) -> Tuple[List[str], List[str]]:
    # returns (taken, remainder)
    if len(tokens) <= max_tokens:
        return tokens, []
    return tokens[:max_tokens], tokens[max_tokens:]

# ---------- PDF extraction & heading detection ----------

def extract_text_and_fonts(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of blocks: {page, text, font_size, font_name, bbox}
    We'll aggregate text with same font-size / font-name on a page as a heuristic for headings.
    """
    doc = fitz.open(pdf_path)
    blocks = []
    for pno in range(len(doc)):
        page = doc[pno]
        try:
            # get dict of text with details
            dict_page = page.get_text("dict")
        except Exception:
            # fallback to simple text
            blocks.append({"page": pno, "text": page.get_text(), "font_size": None, "font_name": None, "bbox": None})
            continue

        # Walk blocks
        for b in dict_page.get("blocks", []):
            if b.get("type") != 0:  # 0=text
                continue
            block_text_parts = []
            font_sizes = []
            font_names = []
            bbox = b.get("bbox")
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    size = span.get("size", None)
                    fname = span.get("font", None)
                    if txt.strip():
                        block_text_parts.append(txt)
                        if size:
                            font_sizes.append(size)
                        if fname:
                            font_names.append(fname)
            if block_text_parts:
                avg_size = sum(font_sizes)/len(font_sizes) if font_sizes else None
                # common font name (most frequent)
                fname = None
                if font_names:
                    fname = max(set(font_names), key=font_names.count)
                blocks.append({
                    "page": pno,
                    "text": " ".join(block_text_parts).strip(),
                    "font_size": avg_size,
                    "font_name": fname,
                    "bbox": bbox
                })
    return blocks

def detect_headings(blocks: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    Heuristic: on each page, compute font_size distribution, mark blocks with font_size >= mean+std or among top sizes as headings.
    """
    from statistics import mean, stdev
    page_groups = {}
    for b in blocks:
        page_groups.setdefault(b["page"], []).append(b)
    out = []
    for pno, blks in page_groups.items():
        sizes = [b["font_size"] for b in blks if b["font_size"] is not None]
        if not sizes:
            # nothing to do
            for b in blks:
                b["is_heading"] = False
                out.append(b)
            continue
        try:
            mu = mean(sizes)
            sigma = stdev(sizes) if len(sizes) > 1 else 0.0
        except Exception:
            mu = mean(sizes)
            sigma = 0.0
        threshold = mu + max(1.0, sigma * 0.5)
        # also consider the top 2 largest sizes as headings
        top_sizes = sorted(set(sizes), reverse=True)[:2]
        for b in blks:
            fs = b["font_size"] or 0
            b["is_heading"] = (fs >= threshold) or (fs in top_sizes)
            out.append(b)
    # preserve original order by page and appearance
    out_sorted = sorted(out, key=lambda x: (x["page"], x["bbox"][1] if x["bbox"] else 0))
    return out_sorted

# ---------- Sectioning ----------

def build_sections_from_blocks(blocks: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    Convert low-level blocks into sections using heading markers.
    Each section: {title, text, start_offset_est, end_offset_est}
    """
    sections = []
    current = {"title": None, "text": []}
    for b in blocks:
        txt = b["text"].strip()
        if b.get("is_heading") and len(txt) < 200:
            # start a new section
            # --- FIX 2 START: Only append section if it has text ---
            if current["text"]: # Only append if there is text
                sections.append(current)
            # --- FIX 2 END ---
            current = {"title": txt, "text": []}
        else:
            current["text"].append(txt)
    # --- FIX 2 START: Only append the last section if it has text ---
    if current["text"]: # Only append if there is text
        sections.append(current)
    # --- FIX 2 END ---

    # flatten text
    for s in sections:
        s["full_text"] = "\n\n".join(s["text"]).strip()
        s["subheading"] = None  # placeholder
        # crude offsets (char offsets not available easily); we'll not compute exact char offsets from PDF
        s["start_offset"] = 0
        s["end_offset"] = len(s["full_text"])
    return sections

# ---------- Chunking ----------

# --- FIX 1 START: Add start_chunk_idx parameter ---
def chunk_section(section: Dict[str,Any],
                  doc_id: str,
                  chunk_min: int = 500,
                  chunk_max: int = 800,
                  overlap_tokens: int = 100,
                  start_chunk_idx: int = 0) -> List[Dict[str,Any]]: # Add start_chunk_idx
# --- FIX 1 END ---
    """
    Chunk a section into 500-800 token blocks, with overlap.
    We operate on sentence boundaries to keep coherence.
    """
    text = section.get("full_text", "").strip()
    if not text:
        return []

    sentences = sentence_split(text)
    # build token-level list by sentences (approx token counts per sentence)
    sent_tokens = []
    for s in sentences:
        cnt = approx_token_count(s)
        sent_tokens.append((s, cnt))

    chunks = []
    i = 0
    total_sents = len(sent_tokens)
    # --- FIX 1 START: Use start_chunk_idx ---
    chunk_idx = start_chunk_idx # Use the starting index
    # --- FIX 1 END ---
    while i < total_sents:
        cur_tokens = 0
        cur_sents = []
        j = i
        # grow until reach at least chunk_min but not exceed chunk_max by too much
        while j < total_sents and (cur_tokens < chunk_min or (cur_tokens < chunk_max and len(cur_sents) == 0)):
            s, c = sent_tokens[j]
            if cur_tokens + c > chunk_max and cur_tokens >= chunk_min:
                break
            cur_sents.append(s)
            cur_tokens += c
            j += 1
        if not cur_sents and j < total_sents: # Ensure j is within bounds before accessing
            # force at least one sentence if possible
            s, c = sent_tokens[j]
            cur_sents.append(s)
            cur_tokens += c
            j += 1
        
        # Only create a chunk if there's actual text
        if not cur_sents:
            break # Avoid infinite loop if no sentence fits

        chunk_text = " ".join(cur_sents).strip()
        chunk_tokens = cur_tokens
        
        # Final check: Don't add extremely small chunks unless it's the only content
        if chunk_tokens < chunk_min / 5 and len(chunks) > 0: # Avoid very small trailing chunks
             pass # Skip this tiny chunk
        else:
            metadata = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_c{chunk_idx:04d}", # Uses incrementing chunk_idx
                "title": section.get("title"),
                "subheading": section.get("subheading"),
                "start_offset": None,
                "end_offset": None,
                "tokens": chunk_tokens,
                "section_tags": [],
                "effective_date": None,
                "jurisdiction": None,
                "sensitive": False,
                "source_citation": doc_id,
                "checksum": sha256_text(chunk_text),
                "text": chunk_text
            }
            chunks.append(metadata)
            chunk_idx += 1

        # advance i by number of sentences used minus overlap in sentences (approx)
        # compute overlap in tokens -> convert to sentences to step back appropriately
        if overlap_tokens <= 0:
            i = j
        else:
            # compute how many sentences correspond to overlap_tokens from the end
            k = j - 1
            overlap_acc = 0
            overlap_sent_count = 0
            while k >= i and overlap_acc < overlap_tokens:
                overlap_acc += sent_tokens[k][1]
                overlap_sent_count += 1
                k -= 1
            # next start is j - overlap_sent_count
            i = max(i + 1, j - overlap_sent_count)  # ensure progress
            
            # Safety break for potential infinite loop if overlap logic prevents progress
            if i >= j and j < total_sents:
                 i = j # Force progress if stuck

    return chunks

# ---------- Special enrichment for cancellations doc ----------

def extract_faq_pairs(text: str) -> List[Dict[str,str]]:
    """
    Heuristic extraction of Q/A style pairs.
    Looks for lines starting with Q: or a sentence ending with '?', then next block(s) as answer.
    """
    lines = [l.strip() for l in re.split(r'\n{1,}', text) if l.strip()]
    qa = []
    i = 0
    while i < len(lines):
        line = lines[i]
        q_match = None
        # startswith Q:, Q., Question, or ends with ?
        if re.match(r'^(Q[:\.\s]|Question[:\.\s])', line, re.I):
            q_text = re.sub(r'^(Q[:\.\s]|Question[:\.\s])', '', line, flags=re.I).strip()
            q_match = q_text
        elif line.endswith('?') and len(line) < 300:
            q_match = line
        if q_match:
            # gather following lines until next question or empty line
            ans_parts = []
            j = i + 1
            while j < len(lines) and not (lines[j].endswith('?') or re.match(r'^(Q[:\.\s]|Question[:\.\s])', lines[j], re.I)):
                ans_parts.append(lines[j])
                j += 1
            answer = " ".join(ans_parts).strip()
            if answer:
                qa.append({"question": q_match, "answer": answer})
            i = j
        else:
            i += 1
    return qa

def extract_addresses(text: str) -> List[Dict[str,str]]:
    """
    Very naive address extraction: looks for keywords 'Address', 'Return to', 'Returns', and 6-digit pincodes.
    Returns list of dicts: {city, address_line, pincode}
    """
    addresses = []
    # find pincode occurrences and nearby text
    for match in re.finditer(r'(\b\d{6}\b)', text):
        pincode = match.group(1)
        span_start = max(0, match.start() - 200)
        span_end = min(len(text), match.end() + 200)
        snippet = text[span_start:span_end]
        # try to find line breaks around snippet
        snippet = snippet.replace('\n', ' ; ')
        # attempt to get city by common patterns e.g. 'City - XYZ' or 'Bengaluru'
        city_match = re.search(r'(Bengaluru|Bangalore|Hyderabad|Kolkata|Mumbai|Delhi|Chennai|Pune|Lucknow|Noida|Gurgaon|Gurugram)', snippet, re.I)
        city = city_match.group(1) if city_match else None
        addresses.append({
            "address_line": snippet.strip(),
            "pincode": pincode,
            "city": city
        })
    # also try keyword-based blocks
    for m in re.finditer(r'(Return(?:s)? (?:address|centre|center|facility)[\:\-]?\s*)(.+?)(?=(?:\n\n)|$)', text, re.I | re.S):
        block = m.group(2).strip()
        pinc = re.search(r'\b\d{6}\b', block)
        addresses.append({
            "address_line": block.replace('\n', ', '),
            "pincode": pinc.group(0) if pinc else None,
            "city": None
        })
    # deduplicate by pincode/address snippet
    seen = set()
    unique = []
    for a in addresses:
        key = (a.get("pincode"), a.get("address_line")[:80])
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique

def extract_non_returnable(text: str) -> List[str]:
    """
    Try to detect phrases like 'non-returnable', 'not eligible', and list category names.
    """
    non_ret = set()
    # look for 'non-returnable' or 'not returnable' and extract list items following colon or dash
    for m in re.finditer(r'(non[- ]returnable|not returnable|non returnable|non-returnable|non refundable|not eligible for return|excluded items)\s*[:\-]?\s*(.+?)(?:\n|$)', text, re.I | re.S):
        candidates = m.group(2)
        # split by commas or semicolons
        for part in re.split(r'[;\,]\s*', candidates):
            item = re.sub(r'[^A-Za-z0-9 &\-\/]', '', part).strip()
            if item and len(item) > 2:
                non_ret.add(item)
    # fallback: look for lists under headings like 'Items not eligible for return'
    for m in re.finditer(r'(Items not eligible for return|Not eligible for returns|Non-returnable items)\s*(?:\n|\:)(.+?)(?=\n\n|$)', text, re.I|re.S):
        block = m.group(2)
        for part in re.split(r'[\n\-\•\*]+', block):
            it = part.strip()
            if it:
                it = re.sub(r'[^A-Za-z0-9 &\-\/]', '', it)
                if len(it) > 2:
                    non_ret.add(it)
    # small curated keywords detection
    keywords = ["innerwear", "intimate", "socks", "earrings", "pierced", "personal hygiene", "food", "perishable", "personalized", "customized", "swimwear"]
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', text, re.I):
            non_ret.add(kw)
    return sorted(list(non_ret))

def extract_refund_timelines(text: str) -> Dict[str,str]:
    """
    Look for patterns like 'credit card' + '7 days', 'UPI' + '2-5 days', etc.
    Returns a dict mapping method -> timeline string
    """
    timelines = {}
    # common methods to look for
    methods = ["credit card", "debit card", "netbanking", "upi", "myntra credit", "wallet", "cash on delivery", "cod", "bank transfer", "razorpay"]
    for m in methods:
        # Adjusted regex to be less strict about surrounding characters and handle variations
        regex = rf'({re.escape(m)})\D{{0,60}}?(\d+\s*(?:to|-|–)?\s*\d*\s*(?:working\s+|business\s+)?days|\binstant(?:ly)?\b|\bimmediate(?:ly)?\b|\bwithin \d+\s*(?:working\s+|business\s+)?days\b)'
        for found in re.finditer(regex, text, re.I):
            timelines[found.group(1).lower()] = found.group(2).strip()
            
    # also generic patterns "refund will be processed within X days for Y"
    for m in re.finditer(r'(\d+\s*(?:to|-|–)?\s*\d*\s*(?:working\s+|business\s+)?days)\s*(?:for|to|via)\s*([\w\s\/]+?)(?:[,\.\n]|$)', text, re.I):
        days = m.group(1).strip()
        method = m.group(2).strip().lower()
        if method not in timelines: # Avoid overwriting more specific matches
             timelines[method] = days
    return timelines

# ---------- Putting it together ----------

def process_pdf_file(pdf_path: str, args) -> List[Dict[str,Any]]:
    filename = os.path.basename(pdf_path)
    # More robust doc_id generation
    doc_id = re.sub(r'\.pdf$', '', filename, flags=re.I) # Remove extension
    doc_id = re.sub(r'[^A-Za-z0-9_\-]+', '_', doc_id).strip('_') # Replace non-alphanum with underscore
    
    print(f"[+] Processing {filename} -> doc_id={doc_id}")

    blocks = extract_text_and_fonts(pdf_path)
    blocks = detect_headings(blocks)
    sections = build_sections_from_blocks(blocks)

    all_chunks = []
    # --- FIX 1 START: Initialize global_chunk_idx ---
    global_chunk_idx = 0 
    # --- FIX 1 END ---
    for sec in sections:
        # --- FIX 1 START: Pass and update global_chunk_idx ---
        sec_chunks = chunk_section(sec, doc_id, 
                                   chunk_min=args.min_tokens, 
                                   chunk_max=args.max_tokens, 
                                   overlap_tokens=args.overlap,
                                   start_chunk_idx=global_chunk_idx) 
        
        global_chunk_idx += len(sec_chunks) # Increment the counter
        # --- FIX 1 END ---
        
        # add section tags heuristics
        for c in sec_chunks:
            title = (c["title"] or "").lower() if c.get("title") else ""
            txt = (c["text"] or "").lower()
            tags = set()
            if any(k in title for k in ["cancel", "cancellation", "modify", "modification", "returns", "refund", "exchange"]):
                tags.update(["cancellations","refund","returns"])
            if "privacy" in title or "data" in txt:
                tags.add("privacy")
            if "grievance" in title or "grievance" in txt:
                tags.add("grievance")
            c["section_tags"] = sorted(list(tags))
            all_chunks.append(c)

    # --- FIX 3 START: Collect document tags ---
    doc_tags = {t for c in all_chunks for t in c.get("section_tags", [])}
    # --- FIX 3 END ---

    # If file looks like cancellations doc OR document tags include relevant terms, run enrichment
    # --- FIX 3 START: Use doc_tags to trigger enrichment ---
    if "cancellations" in doc_tags or "refund" in doc_tags or "returns" in doc_tags:
    # --- FIX 3 END ---
        full_text = "\n\n".join([s.get("full_text","") for s in sections if s.get("full_text")]) # Ensure text exists
        if full_text: # Only run if there is text to process
            print(f"[*] Performing enrichment for {doc_id} based on tags: {doc_tags}")
            faq_pairs = extract_faq_pairs(full_text)
            if faq_pairs: # Only add if non-empty
                 add_special_chunk(all_chunks, doc_id, "cancellations_faq_pairs", json.dumps(faq_pairs, ensure_ascii=False, indent=2), tags=["cancellations","faq","actionable"], extra={"qa_count": len(faq_pairs)})
            
            addresses = extract_addresses(full_text)
            if addresses: # Only add if non-empty
                 add_special_chunk(all_chunks, doc_id, "cancellations_return_addresses", json.dumps(addresses, ensure_ascii=False, indent=2), tags=["cancellations","returns","addresses"], extra={"address_count": len(addresses)})

            non_returnable = extract_non_returnable(full_text)
            if non_returnable: # Only add if non-empty
                 add_special_chunk(all_chunks, doc_id, "cancellations_non_returnable", json.dumps(non_returnable, ensure_ascii=False, indent=2), tags=["cancellations","non_returnable"], extra={"non_returnable_count": len(non_returnable)})

            timelines = extract_refund_timelines(full_text)
            if timelines: # Only add if non-empty
                 add_special_chunk(all_chunks, doc_id, "cancellations_refund_timelines", json.dumps(timelines, ensure_ascii=False, indent=2), tags=["cancellations","refund","timeline"], extra={"timeline_count": len(timelines)})
            
            # add a decision_rules chunk (high-level rules from heuristics)
            decision_rules_text = (
                "Decision rules (auto-generated):\n"
                "- If order status has 'cancel' button visible in UI -> allow cancel immediately.\n"
                "- If order status is 'packed' and cancel button not available -> instruct user to refuse delivery; mark for pickup return.\n"
                "- If prepaid -> refund to original source using refund_timelines mapping.\n"
                "- If COD -> initiate payout to wallet or bank within the refund timeline.\n"
            )
            add_special_chunk(all_chunks, doc_id, "cancellations_decision_rules", decision_rules_text, tags=["cancellations","decision_rules","actionable"], extra={})
    
    # Filter out any potentially remaining very small chunks if they aren't special chunks
    final_chunks = [c for c in all_chunks if c['tokens'] > args.min_tokens / 10 or (c['title'] and c['title'].startswith("cancellations_"))] # Keep special chunks regardless of size
    
    return final_chunks # Return the filtered list

def add_special_chunk(all_chunks, doc_id, chunk_key, text, tags=None, extra=None):
    # Use a simpler, unique ID for special chunks based on key
    chunk_id = f"{doc_id}_{chunk_key}"
    
    # Check if this chunk ID already exists to prevent duplicates if script runs weirdly
    if any(c['chunk_id'] == chunk_id for c in all_chunks):
         print(f"[!] Warning: Special chunk {chunk_id} already exists. Skipping.")
         return None # Indicate skipped

    c = {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "title": chunk_key,
        "subheading": None,
        "start_offset": None,
        "end_offset": None,
        "tokens": approx_token_count(text),
        "section_tags": tags or [],
        "effective_date": None,
        "jurisdiction": None,
        "sensitive": False,
        "source_citation": doc_id,
        "checksum": sha256_text(text),
        "text": text
    }
    if extra is not None: # Check for None explicitly
        c.update({"extra": extra})
    all_chunks.append(c)
    return c

# ---------- CLI / Runner ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="./", help="Directory containing PDFs")
    parser.add_argument("--output", default="./chunks_out", help="Output directory")
    parser.add_argument("--min-tokens", type=int, default=50) # Lowered min default slightly
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    args = parser.parse_args()

    # Input validation
    input_path = Path(args.input_dir)
    if not input_path.is_dir():
         print(f"[Error] Input directory not found: {args.input_dir}")
         return

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "chunks.jsonl"
    
    pdf_files = [str(p) for p in input_path.glob("*.pdf")]
    if not pdf_files:
        print(f"No PDFs found in input dir: {args.input_dir}")
        return

    all_chunks_total = []
    for pdf in pdf_files:
        try:
            chunks = process_pdf_file(pdf, args)
            all_chunks_total.extend(chunks) # Use extend for list addition
        except Exception as e:
            print(f"[Error] Failed to process {pdf}: {e}")
            # Optionally continue to next file or re-raise based on desired robustness

    if not all_chunks_total:
         print("[!] No chunks generated.")
         return

    # write jsonl
    try:
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for c in all_chunks_total:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    except IOError as e:
         print(f"[Error] Failed to write chunks to {jsonl_path}: {e}")
         return

    # create a small index file summarizing counts and tags
    try:
        by_doc = {}
        for c in all_chunks_total:
            by_doc.setdefault(c["doc_id"], []).append(c)
        index_summary = {doc: {"chunk_count": len(chs),
                               "tags": sorted(list({t for ch in chs for t in (ch.get("section_tags") or [])}))}
                         for doc, chs in by_doc.items()}
        summary_path = output_path / "index_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(index_summary, fh, indent=2, ensure_ascii=False)
        print(f"[+] Wrote index summary to {summary_path}")
    except Exception as e:
         print(f"[Error] Failed to create index summary: {e}")


    print(f"[+] Wrote {len(all_chunks_total)} chunks to {jsonl_path}")


if __name__ == "__main__":
    main()