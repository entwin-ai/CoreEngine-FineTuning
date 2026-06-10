"""
watch.py — INGEST TRIGGER (the new front door to the pipeline)

Watches a HARDCODED inbox folder. Whenever new files arrive it:
  1. waits until each file is fully written (size-stable — avoids parsing half-copied files),
  2. parses it into authored stylistic units (file_parser.parse_file),
  3. appends them to data/raw_messages.jsonl (de-duplicated),
  4. after a short debounce (so a burst of files is handled as one batch), triggers the
     downstream pipeline: profile -> build_dataset -> [train -> evaluate -> merge].

Two backends, auto-selected:
  - watchdog (real OS filesystem events) if installed  -> instant, efficient
  - polling fallback (no dependencies)                 -> scans every POLL_SECONDS

A processed-file ledger (data/.processed.json) records file path + mtime + size so restarts
never reprocess the same file, and an edited/re-dropped file IS reprocessed.

Run:
    python scripts/watch.py                  # watch + auto-run profile/build on each batch
    python scripts/watch.py --full           # also run train+evaluate+merge after each batch
    python scripts/watch.py --once           # process whatever is already in the folder, exit
    ENTWIN_INBOX=/path python scripts/watch.py   # override the hardcoded folder
"""
import os, sys, json, time, hashlib, subprocess, threading
import file_parser
import monotonic_id

# ----------------- HARDCODED inbox folder (override via env if needed) -----------------
INBOX = os.environ.get("ENTWIN_INBOX", os.path.expanduser("~/entwin_inbox"))
# Dedicated HARDCODED folder where files go AFTER reading (separate from the inbox).
PROCESSED_DIR = os.environ.get("ENTWIN_PROCESSED", os.path.expanduser("~/entwin_processed"))
RAW = "data/raw_messages.jsonl"
LEDGER = "data/.processed.json"
STABLE_CHECKS = 2          # consecutive equal size readings => file fully written
STABLE_INTERVAL = 0.8      # seconds between size checks
POLL_SECONDS = 3.0         # polling-backend scan interval
DEBOUNCE_SECONDS = 4.0     # wait for a burst to settle before running the pipeline
RUN_FULL = "--full" in sys.argv

SUPPORTED = (".txt", ".md", ".json", ".jsonl", ".eml", ".csv",
             ".html", ".htm", ".pdf", ".docx")

# ----------------- ledger -----------------
def _load_ledger():
    try:
        return json.load(open(LEDGER))
    except Exception:
        return {}

def _save_ledger(d):
    json.dump(d, open(LEDGER, "w"))

def _sig(path):
    st = os.stat(path)
    return f"{int(st.st_mtime)}:{st.st_size}"

# ----------------- file readiness -----------------
def _wait_stable(path):
    last, stable = -1, 0
    for _ in range(60):  # up to ~48s
        try:
            size = os.path.getsize(path)
        except OSError:
            return False
        if size == last and size > 0:
            stable += 1
            if stable >= STABLE_CHECKS:
                return True
        else:
            stable = 0
        last = size
        time.sleep(STABLE_INTERVAL)
    return True  # proceed anyway after timeout

# ----------------- corpus append (deduped) -----------------
_append_lock = threading.Lock()
def _append_rows(rows):
    if not rows:
        return 0
    os.makedirs("data", exist_ok=True)
    existing = set()
    if os.path.exists(RAW):
        for line in open(RAW, encoding="utf-8"):
            try:
                existing.add(_text_key(json.loads(line)["text"]))
            except Exception:
                pass
    added = 0
    with _append_lock, open(RAW, "a", encoding="utf-8") as f:
        for r in rows:
            k = _text_key(r["text"])
            if k in existing:
                continue
            existing.add(k)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            added += 1
    return added

def _text_key(t):
    return hashlib.sha1(" ".join(t.split()).lower().encode()).hexdigest()

# ----------------- process one file -----------------
def process_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED:
        return 0
    ledger = _load_ledger()
    sig = _sig(path)
    if ledger.get(path) == sig:
        return 0  # already processed this exact version
    if not _wait_stable(path):
        return 0
    rows = file_parser.parse_file(path)
    added = _append_rows(rows)
    ledger[path] = _sig(path)
    _save_ledger(ledger)
    moved = _archive(path)   # reading complete -> move to processed folder with sortable prefix
    where = f" -> {os.path.basename(moved)}" if moved else ""
    print(f"[ingest] {os.path.basename(path)}: parsed {len(rows)} units, +{added} new{where}")
    return added

def _archive(path):
    """Move a fully-read file into the hardcoded processed folder, renamed with a
    monotonically-increasing alphanumeric prefix so that sorting by name == chronological order.
    Returns the new path, or None if the move was skipped/failed."""
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        uid = monotonic_id.next_id()                 # e.g. "0abc12xyz-0000"
        base = os.path.basename(path)
        dst = os.path.join(PROCESSED_DIR, f"{uid}__{base}")
        # guard against an impossibly-rare same-name collision
        while os.path.exists(dst):
            uid = monotonic_id.next_id()
            dst = os.path.join(PROCESSED_DIR, f"{uid}__{base}")
        os.replace(path, dst)                        # atomic on same filesystem
        return dst
    except OSError:
        # cross-device move (inbox and processed on different mounts) -> copy + remove
        try:
            import shutil
            shutil.move(path, dst)
            return dst
        except Exception as e:
            print(f"[archive skip] {e}")
            return None
    except Exception as e:
        print(f"[archive skip] {e}")
        return None

# ----------------- downstream pipeline -----------------
def run_pipeline():
    here = os.path.dirname(os.path.abspath(__file__))
    steps = [("profile.py", "measure fingerprint + guardrail"),
             ("build_dataset.py", "rebuild style-transfer dataset")]
    if RUN_FULL:
        steps += [("train_qlora.py", "QLoRA fine-tune"),
                  ("evaluate.py", "drift + fact-leak eval"),
                  ("merge_serve.py", "merge + export")]
    for script, desc in steps:
        print(f"  -> {desc} ({script})")
        rc = subprocess.run([sys.executable, os.path.join(here, script)],
                            cwd=os.path.dirname(here)).returncode
        if rc != 0:
            print(f"  [stop] {script} exited {rc}; fix before continuing.")
            return
    print("[pipeline] batch complete.")

# ----------------- debounced trigger -----------------
_timer = None
_pending = threading.Lock()
def schedule_pipeline():
    global _timer
    with _pending:
        if _timer:
            _timer.cancel()
        _timer = threading.Timer(DEBOUNCE_SECONDS, run_pipeline)
        _timer.daemon = True
        _timer.start()

# ----------------- backends -----------------
def scan_existing():
    found = 0
    for name in sorted(os.listdir(INBOX)):
        path = os.path.join(INBOX, name)
        if os.path.isfile(path):
            found += 1 if process_file(path) else 0
    return found

def run_polling():
    print(f"[watch] polling {INBOX} every {POLL_SECONDS}s "
          f"({'FULL pipeline' if RUN_FULL else 'profile+dataset'} per batch)")
    seen_sig = {}
    while True:
        changed = False
        try:
            names = os.listdir(INBOX)
        except FileNotFoundError:
            os.makedirs(INBOX, exist_ok=True); names = []
        for name in sorted(names):
            path = os.path.join(INBOX, name)
            if not os.path.isfile(path):
                continue
            try:
                sig = _sig(path)
            except OSError:
                continue
            if seen_sig.get(path) != sig:
                seen_sig[path] = sig
                if process_file(path):
                    changed = True
        if changed:
            schedule_pipeline()
        time.sleep(POLL_SECONDS)

def run_watchdog():
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class Handler(FileSystemEventHandler):
        def _handle(self, path):
            if os.path.isfile(path) and process_file(path):
                schedule_pipeline()
        def on_created(self, e):
            if not e.is_directory:
                self._handle(e.src_path)
        def on_modified(self, e):
            if not e.is_directory:
                self._handle(e.src_path)
        def on_moved(self, e):
            if not e.is_directory:
                self._handle(e.dest_path)

    print(f"[watch] watchdog on {INBOX} "
          f"({'FULL pipeline' if RUN_FULL else 'profile+dataset'} per batch)")
    obs = Observer()
    obs.schedule(Handler(), INBOX, recursive=False)
    obs.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()

# ----------------- main -----------------
def main():
    os.makedirs(INBOX, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    print(f"[watch] inbox = {INBOX}")

    pre = scan_existing()          # handle anything already sitting in the folder
    if pre:
        print(f"[watch] processed {pre} pre-existing file(s)")
        run_pipeline()

    if "--once" in sys.argv:
        return

    try:
        run_watchdog()
    except ImportError:
        run_polling()

if __name__ == "__main__":
    main()
