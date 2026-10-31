"""Monitoring log (SQLite). Stores metrics only, never the images."""
import re
import sqlite3
from datetime import datetime, timezone

PID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def connect(path):
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS records(
        id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL, cancer TEXT NOT NULL,
        pred_class TEXT NOT NULL, confidence REAL NOT NULL, attn_area REAL NOT NULL, created_at TEXT NOT NULL)""")
    con.execute("CREATE INDEX IF NOT EXISTS ix_pc ON records(patient_id, cancer)")
    return con


def last_record(con, pid, cancer):
    r = con.execute("SELECT * FROM records WHERE patient_id=? AND cancer=? ORDER BY id DESC LIMIT 1", (pid, cancer)).fetchone()
    return dict(r) if r else None


def add(con, pid, cancer, pred, conf, area):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.execute("INSERT INTO records(patient_id,cancer,pred_class,confidence,attn_area,created_at) VALUES(?,?,?,?,?,?)",
                (pid, cancer, pred, conf, area, ts))
    con.commit()


def history(con, pid, cancer=None, limit=50):
    q, a = "SELECT * FROM records WHERE patient_id=?", [pid]
    if cancer:
        q += " AND cancer=?"; a.append(cancer)
    return [dict(r) for r in con.execute(q + " ORDER BY id DESC LIMIT ?", (*a, limit)).fetchall()]


def delete(con, pid):
    n = con.execute("DELETE FROM records WHERE patient_id=?", (pid,)).rowcount
    con.commit()
    return n
