"""
Step 1 — EXTRACT
Pull ONLY messages the user authored (sent mail, sent WhatsApp, own Drive docs).
Style fine-tuning must learn from the user's own output, never from what they received.

Outputs data/raw_messages.jsonl with one record per message:
  {"id","source","text","ts","recipient_hint","is_reply","thread_id"}

Auth: reuses the Gmail OAuth flow you already set up for ingestion (token.json).
WhatsApp + Drive paths are local-file based so nothing leaves your machine.
"""
import os, json, base64, re, glob
from email import message_from_bytes
from datetime import datetime

OUT = "data/raw_messages.jsonl"
USER_EMAIL = os.environ.get("ENTWIN_USER_EMAIL", "hitghosh@gmail.com")

# ---------- GMAIL (sent only) ----------
def extract_gmail(max_messages=8000):
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials.from_authorized_user_file("token.json",
              ["https://www.googleapis.com/auth/gmail.readonly"])
    svc = build("gmail", "v1", credentials=creds)
    out, page = [], None
    while len(out) < max_messages:
        resp = svc.users().messages().list(
            userId="me", q="in:sent -in:chats", maxResults=500, pageToken=page).execute()
        for m in resp.get("messages", []):
            full = svc.users().messages().get(userId="me", id=m["id"], format="raw").execute()
            raw = base64.urlsafe_b64decode(full["raw"].encode())
            msg = message_from_bytes(raw)
            body = _plain_body(msg)
            body = _strip_quoted_and_signature(body)
            if not _is_usable(body):
                continue
            out.append({
                "id": m["id"], "source": "gmail",
                "text": body,
                "ts": int(full.get("internalDate", "0")) // 1000,
                "recipient_hint": _hash_recipient(msg.get("To", "")),
                "is_reply": bool(msg.get("In-Reply-To")),
                "thread_id": full.get("threadId", ""),
            })
        page = resp.get("nextPageToken")
        if not page:
            break
    return out

def _plain_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                try:
                    return part.get_payload(decode=True).decode(errors="ignore")
                except Exception:
                    pass
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                from bs4 import BeautifulSoup
                html = part.get_payload(decode=True).decode(errors="ignore")
                return BeautifulSoup(html, "html.parser").get_text("\n")
        return ""
    try:
        return msg.get_payload(decode=True).decode(errors="ignore")
    except Exception:
        return msg.get_payload() or ""

# ---------- WHATSAPP (exported .txt, sender == user) ----------
WA_LINE = re.compile(r"^\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s+[\d:apmAPM\s]+\]?\s*-?\s*([^:]+):\s(.*)$")
def extract_whatsapp(export_dir="data/whatsapp_exports", my_names=("Nishit", "Nishit Ghosh")):
    out = []
    for path in glob.glob(os.path.join(export_dir, "*.txt")):
        thread = os.path.basename(path)
        buff, cur = [], None
        for line in open(path, encoding="utf-8", errors="ignore"):
            mt = WA_LINE.match(line.strip())
            if mt:
                if cur and cur["sender"] in my_names:
                    _flush_wa(out, cur, thread)
                cur = {"sender": mt.group(2).strip(), "text": mt.group(3)}
            elif cur:
                cur["text"] += "\n" + line.rstrip("\n")
        if cur and cur["sender"] in my_names:
            _flush_wa(out, cur, thread)
    return out

def _flush_wa(out, cur, thread):
    t = _strip_quoted_and_signature(cur["text"])
    if _is_usable(t, min_words=4):
        out.append({"id": f"wa_{len(out)}", "source": "whatsapp", "text": t,
                    "ts": 0, "recipient_hint": _hash_str(thread),
                    "is_reply": True, "thread_id": thread})

# ---------- DRIVE (local exported .txt/.md the user wrote) ----------
def extract_drive(docs_dir="data/drive_docs"):
    out = []
    for path in glob.glob(os.path.join(docs_dir, "**/*.*"), recursive=True):
        if not path.lower().endswith((".txt", ".md")):
            continue
        text = open(path, encoding="utf-8", errors="ignore").read()
        # Split long docs into paragraph-sized stylistic units
        for i, para in enumerate(re.split(r"\n\s*\n", text)):
            p = para.strip()
            if _is_usable(p, min_words=15):
                out.append({"id": f"drive_{os.path.basename(path)}_{i}", "source": "drive",
                            "text": p, "ts": 0, "recipient_hint": "doc",
                            "is_reply": False, "thread_id": os.path.basename(path)})
    return out

# ---------- cleaning helpers ----------
SIG_MARKERS = ("sent from my", "get outlook", "--", "thanks,", "regards,", "best,", "nishit")
def _strip_quoted_and_signature(text):
    lines, kept = text.splitlines(), []
    for ln in lines:
        s = ln.strip()
        if s.startswith(">") or s.startswith("On ") and "wrote:" in s:
            break  # quoted reply chain begins
        if re.match(r"^_{3,}$|^-{3,}$", s):
            break  # signature divider
        kept.append(ln)
    body = "\n".join(kept).strip()
    # drop trailing signature block but KEEP closings (closings are Voice & Identity signal)
    return body

def _is_usable(text, min_words=8, max_words=400):
    if not text:
        return False
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    # drop near-empty / link-only / forwarded boilerplate
    if text.lower().count("http") > 3:
        return False
    if re.search(r"unsubscribe|view in browser|do not reply", text, re.I):
        return False
    return True

import hashlib
def _hash_str(s):  return hashlib.sha1(s.encode()).hexdigest()[:8]
def _hash_recipient(to):  return _hash_str(re.sub(r"[<>\"]", "", to).lower())[:8]

def main():
    os.makedirs("data", exist_ok=True)
    rows = []
    try:
        rows += extract_gmail()
        print(f"gmail: {len(rows)}")
    except Exception as e:
        print(f"[skip gmail] {e}")
    rows += extract_whatsapp();  print(f"+whatsapp -> {len(rows)}")
    rows += extract_drive();     print(f"+drive -> {len(rows)}")
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} authored messages -> {OUT}")

if __name__ == "__main__":
    main()
