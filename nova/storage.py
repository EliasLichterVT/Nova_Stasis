import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone


class Storage:
    def __init__(self, path):
        self.path = path
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    kind TEXT NOT NULL, id TEXT NOT NULL, created TEXT NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY (kind, id)
                )
            """)

    def put(self, kind, record_id, payload):
        created = datetime.now(timezone.utc).isoformat()
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("""
                INSERT INTO records VALUES (?, ?, ?, ?)
                ON CONFLICT(kind, id) DO UPDATE SET payload=excluded.payload
            """, (kind, str(record_id), created, json.dumps(payload)))

    def get(self, kind, record_id):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT payload FROM records WHERE kind=? AND id=?",
                             (kind, str(record_id))).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, kind, limit):
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute("SELECT payload FROM records WHERE kind=? ORDER BY created DESC LIMIT ?",
                              (kind, limit)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete(self, kind, record_id):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, str(record_id)))
