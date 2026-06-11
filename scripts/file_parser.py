"""
file_parser.py — parse ONE incoming file into authored stylistic units.

Called by watch.py when a new file lands in the inbox. Returns a list of records in the
SAME schema extract.py emits, so everything downstream (profile -> build_dataset -> train)
works unchanged:
    {"id","source","text","ts","recipient_hint","is_reply","thread_id"}

Supported formats (detected by extension, falling back to plain text):
    .txt .md            -> paragraph-split
    .json .jsonl        -> if already in message schema, pass through; else pull "text"/"body"
    .eml                -> RFC822 email: take plain body, strip quoted chain + boilerplate
    .csv                -> a column named text/body/message/content is the authored text
    .html .htm          -> visible text, paragraph-split
    .pdf                -> text layer via pdfminer (best-effort; skips scanned/imageonly)
    .docx               -> paragraphs via python-docx
    WhatsApp export .txt -> auto-detected by line pattern, keeps only the user's own lines

Reuses extract.py's cleaning helpers so the SAME quality bar applies to streamed files as to
the initial bulk pull (strip signatures/quotes, drop boilerplate, enforce length window).
"""
import os, re, json, hashlib, time
import extract  # reuse _is_usable, _strip_quoted_and_signature, _hash_str, WA_LINE

MY_NAMES = tuple(n.strip() for n in
                 os.environ.get("ENTWIN_MY_NAMES", "Nishit,Nishit Ghosh,Nishit K Ghosh").split(","))

def parse_file(path):
    ext = os.path.splitext(path)[1].lower()
    ts = int(os.path.getmtime(path))
    thread = os.path.basename(path)
    try:
        if ext in (".txt", ".md"):
            units = _parse_text_whatsapp_or_transcript(path)
        elif ext in (".vtt", ".srt"):
            units = _parse_caption_transcript(path)
        elif ext == ".jsonl":
            units = _parse_jsonl(path)
        elif ext == ".json":
            units = _parse_json(path)
        elif ext == ".eml":
            units = _parse_eml(path)
        elif ext == ".csv":
            units = _parse_csv(path)
        elif ext in (".html", ".htm"):
            units = _parse_html(path)
        elif ext == ".pdf":
            units = _parse_pdf(path)
        elif ext == ".docx":
            units = _parse_docx(path)
        else:
            units = _parse_text_whatsapp_or_transcript(path)  # best-effort plain text
    except Exception as e:
        print(f"[parse error] {path}: {e}")
        return []

    rows = []
    for i, (text, is_reply) in enumerate(units):
        text = extract._strip_quoted_and_signature(text)
        if not extract._is_usable(text):
            continue
        rows.append({
            "id": f"{_hash_path(path)}_{i}",
            "source": f"inbox:{ext.lstrip('.') or 'txt'}",
            "text": text,
            "ts": ts,
            "recipient_hint": extract._hash_str(thread),
            "is_reply": is_reply,
            "thread_id": thread,
        })
    return rows

# ---------- per-format parsers: each yields (text, is_reply) tuples ----------
# A "speaker:" line at the start of a turn, e.g. "Nishit: ..." or "[00:12] Nishit Ghosh: ..."
_TRANSCRIPT_LINE = re.compile(
    r"^\s*(?:\[?\d{1,2}:\d{2}(?::\d{2})?\]?\s*)?([A-Z][\w .'-]{1,40}):\s+(.*)$")

def _parse_text_whatsapp_or_transcript(path):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    lines = raw.splitlines()
    # WhatsApp export detection: timestamped "date - sender: msg" pattern
    wa_hits = sum(1 for ln in lines[:50] if extract.WA_LINE.match(ln.strip()))
    if wa_hits >= 5:
        return _whatsapp_units(raw)
    # Transcript detection: many lines look like "Speaker: text" with 2+ distinct speakers
    tr_hits = [m for ln in lines[:80] if (m := _TRANSCRIPT_LINE.match(ln))]
    speakers = {m.group(1).strip() for m in tr_hits}
    if len(tr_hits) >= 3 and len(speakers) >= 2:
        return _transcript_units(lines)
    # plain prose
    return [(p.strip(), False) for p in re.split(r"\n\s*\n", raw) if p.strip()]

def _transcript_units(lines):
    """Speaker-aware: keep ONLY the author's turns (sender in MY_NAMES). A meeting transcript
    contains many voices; ingesting all of them teaches the twin other people's words."""
    units, cur = [], None
    for line in lines:
        m = _TRANSCRIPT_LINE.match(line)
        if m:
            if cur and _is_author(cur["speaker"]):
                units.append((cur["text"].strip(), True))
            cur = {"speaker": m.group(1).strip(), "text": m.group(2)}
        elif cur:
            cur["text"] += " " + line.strip()
    if cur and _is_author(cur["speaker"]):
        units.append((cur["text"].strip(), True))
    return units

def _parse_caption_transcript(path):
    """Parse .vtt / .srt caption files. These often carry speaker tags like '<v Nishit>...'
    (WEBVTT) or 'Nishit: ...' inline. Without speaker tags we cannot attribute turns, so we
    keep author-tagged cues only; if none are tagged, we skip (can't safely attribute)."""
    raw = open(path, encoding="utf-8", errors="ignore").read()
    # strip indices, timestamps, and WEBVTT header
    cues, cur_speaker, buf = [], None, []
    voice = re.compile(r"<v\s+([^>]+)>(.*?)(?:</v>|$)", re.I)
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.upper().startswith("WEBVTT") or "-->" in s or s.isdigit():
            continue
        vm = voice.search(s)
        if vm:
            spk, txt = vm.group(1).strip(), re.sub(r"<[^>]+>", "", vm.group(2)).strip()
            cues.append((spk, txt))
        else:
            im = _TRANSCRIPT_LINE.match(s)
            if im:
                cues.append((im.group(1).strip(), im.group(2).strip()))
    # merge consecutive cues by same speaker, keep only the author's
    units, cur = [], None
    for spk, txt in cues:
        if cur and cur["speaker"] == spk:
            cur["text"] += " " + txt
        else:
            if cur and _is_author(cur["speaker"]):
                units.append((cur["text"].strip(), True))
            cur = {"speaker": spk, "text": txt}
    if cur and _is_author(cur["speaker"]):
        units.append((cur["text"].strip(), True))
    return units

def _is_author(name):
    n = (name or "").strip().lower()
    return any(n == mn.lower() or mn.lower() in n for mn in MY_NAMES)

def _whatsapp_units(raw):
    units, cur = [], None
    for line in raw.splitlines():
        mt = extract.WA_LINE.match(line.strip())
        if mt:
            if cur and cur["sender"] in MY_NAMES:
                units.append((cur["text"], True))
            cur = {"sender": mt.group(2).strip(), "text": mt.group(3)}
        elif cur:
            cur["text"] += "\n" + line
    if cur and cur["sender"] in MY_NAMES:
        units.append((cur["text"], True))
    return units

def _parse_jsonl(path):
    out = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        out += _from_obj(obj)
    return out

def _parse_json(path):
    data = json.load(open(path, encoding="utf-8", errors="ignore"))
    items = data if isinstance(data, list) else [data]
    out = []
    for obj in items:
        out += _from_obj(obj)
    return out

def _from_obj(obj):
    if not isinstance(obj, dict):
        return [(str(obj), False)] if str(obj).strip() else []
    # already in our message schema?
    if "text" in obj and "source" in obj:
        return [(obj["text"], bool(obj.get("is_reply", False)))]
    for key in ("text", "body", "message", "content"):
        if key in obj and isinstance(obj[key], str):
            return [(obj[key], bool(obj.get("is_reply", False)))]
    return []

def _parse_eml(path):
    from email import message_from_bytes
    msg = message_from_bytes(open(path, "rb").read())
    body = extract._plain_body(msg)
    is_reply = bool(msg.get("In-Reply-To"))
    return [(body, is_reply)] if body.strip() else []

def _parse_csv(path):
    import csv
    out = []
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        col = None
        for cand in ("text", "body", "message", "content"):
            if reader.fieldnames and cand in [c.lower() for c in reader.fieldnames]:
                col = next(c for c in reader.fieldnames if c.lower() == cand)
                break
        if not col:
            return []
        for row in reader:
            v = (row.get(col) or "").strip()
            if v:
                out.append((v, False))
    return out

def _parse_html(path):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(open(path, encoding="utf-8", errors="ignore").read(), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    return [(p.strip(), False) for p in re.split(r"\n\s*\n", text) if p.strip()]

def _parse_pdf(path):
    try:
        from pdfminer.high_level import extract_text
    except ImportError:
        print("[pdf] pip install pdfminer.six to enable PDF parsing"); return []
    text = extract_text(path) or ""
    return [(p.strip(), False) for p in re.split(r"\n\s*\n", text) if p.strip()]

def _parse_docx(path):
    try:
        import docx
    except ImportError:
        print("[docx] pip install python-docx to enable .docx parsing"); return []
    d = docx.Document(path)
    paras = [p.text for p in d.paragraphs if p.text.strip()]
    # group consecutive paragraphs into stylistic units (~ message-sized)
    return [(p.strip(), False) for p in paras]

def _hash_path(path):
    return hashlib.sha1((path + str(os.path.getmtime(path))).encode()).hexdigest()[:10]

if __name__ == "__main__":
    import sys
    rows = parse_file(sys.argv[1])
    print(f"parsed {len(rows)} usable units from {sys.argv[1]}")
    for r in rows[:3]:
        print(" -", r["text"][:100].replace("\n", " "))
