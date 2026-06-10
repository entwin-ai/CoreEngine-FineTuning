"""
monotonic_id.py — generate alphanumeric identifiers that ALWAYS grow over time, so that
sorting filenames as plain strings reproduces the chronological order they were created.

Design
------
An ID is: <TIME_36>-<SEQ_36>

  TIME_36 : current time in milliseconds since epoch, encoded base36, ZERO-PADDED to a fixed
            width. Base36 (0-9a-z) is alphanumeric and, when zero-padded to constant width,
            sorts lexicographically in the SAME order as numerically. 9 base36 chars cover
            36^9 ms ≈ until year 3290, so width is safe and fixed for ~millennia.

  SEQ_36  : a per-instance counter, base36, zero-padded to 4 chars. Breaks ties when several
            files are processed within the same millisecond, preserving issue order.

Two guarantees make it strictly increasing:
  1. The last issued (time, seq) pair is persisted to a small state file. If the wall clock
     ever moves backwards (NTP correction, restart on a laggy clock), we clamp forward to
     last_time and bump seq, so the ID never decreases.
  2. Within a millisecond, seq increments; when the clock advances, seq resets to 0.

Because every field is fixed-width and uses the alphabet 0-9a-z (which is already in
ascending ASCII order), a simple `sorted(filenames)` yields chronological order.
"""
import os, time, json, threading

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"  # base36, ascending ASCII order
TIME_WIDTH = 9      # 36^9 ms ≈ year 3290; fixed width keeps lexicographic == numeric
SEQ_WIDTH = 4       # up to 36^4 = 1,679,616 ids per millisecond
STATE_FILE = os.environ.get("ENTWIN_ID_STATE", "data/.idstate.json")

_lock = threading.Lock()


def _to_base36(n, width):
    if n < 0:
        raise ValueError("negative")
    s = ""
    while n:
        n, r = divmod(n, 36)
        s = ALPHABET[r] + s
    s = s or "0"
    if len(s) > width:
        raise OverflowError(f"value needs {len(s)} chars, width is {width}")
    return s.rjust(width, "0")


def _load_state():
    try:
        with open(STATE_FILE) as f:
            d = json.load(f)
            return int(d.get("last_ms", 0)), int(d.get("last_seq", -1))
    except Exception:
        return 0, -1


def _save_state(ms, seq):
    os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"last_ms": ms, "last_seq": seq}, f)
    os.replace(tmp, STATE_FILE)  # atomic


def next_id():
    """Return a new ID strictly greater (lexicographically) than every prior one."""
    with _lock:
        now = int(time.time() * 1000)
        last_ms, last_seq = _load_state()
        if now > last_ms:
            ms, seq = now, 0
        else:
            # clock didn't advance (same ms) or went backwards -> clamp forward, bump seq
            ms, seq = last_ms, last_seq + 1
            if seq >= 36 ** SEQ_WIDTH:        # overflow the seq -> roll into next ms
                ms, seq = last_ms + 1, 0
        _save_state(ms, seq)
        return f"{_to_base36(ms, TIME_WIDTH)}-{_to_base36(seq, SEQ_WIDTH)}"


if __name__ == "__main__":
    # quick self-check: a rapid burst must come out strictly increasing AND sorted-equal
    ids = [next_id() for _ in range(20)]
    assert ids == sorted(ids), "IDs not lexicographically sorted!"
    assert len(set(ids)) == len(ids), "IDs not unique!"
    print("OK strictly increasing & unique. sample:")
    for i in ids[:5]:
        print(" ", i)
