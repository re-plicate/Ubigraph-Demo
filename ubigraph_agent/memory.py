"""Persistent conversation and task-event memory; separate from the GNN label bank."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid


def uid():
    return uuid.uuid4().hex


class MemoryStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.lock = path, threading.RLock()
        with self.db() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,title TEXT,context TEXT,created REAL,updated REAL);
              CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT,session_id TEXT,role TEXT,content TEXT,task_id TEXT,created REAL);
              CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,session_id TEXT,status TEXT,context TEXT,error TEXT,created REAL,updated REAL);
              CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT,type TEXT,data TEXT,created REAL);
              CREATE INDEX IF NOT EXISTS event_task ON events(task_id,id);
            """)
            db.execute("UPDATE tasks SET status='interrupted',error='服务重启，任务已中断' WHERE status IN ('queued','running')")

    @contextmanager
    def db(self):
        with self.lock:
            db = sqlite3.connect(self.path)
            db.row_factory = sqlite3.Row
            try:
                yield db
                db.commit()
            finally:
                db.close()

    def create_session(self, context):
        identity, now = uid(), time.time()
        with self.db() as db:
            db.execute("INSERT INTO sessions VALUES(?,?,?,?,?)", (identity, "新会话", json.dumps(context), now, now))
        return self.session(identity)

    def session(self, identity):
        with self.db() as db:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise ValueError("会话不存在")
        result = dict(row)
        result["context"] = json.loads(result["context"])
        return result

    def sessions(self):
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT id,title,updated,context FROM sessions ORDER BY updated DESC LIMIT 30")]

    def message(self, session_id, role, content, task_id):
        now = time.time()
        with self.db() as db:
            db.execute("INSERT INTO messages(session_id,role,content,task_id,created) VALUES(?,?,?,?,?)",
                       (session_id, role, content, task_id, now))
            db.execute("UPDATE sessions SET updated=? WHERE id=?", (now, session_id))
            if role == "user":
                db.execute("UPDATE sessions SET title=? WHERE id=? AND title='新会话'", (content[:32], session_id))

    def history(self, session_id):
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM messages WHERE session_id=? ORDER BY id", (session_id,))]

    def create_task(self, session_id, context):
        identity, now = uid(), time.time()
        with self.db() as db:
            if db.execute("SELECT id FROM tasks WHERE session_id=? AND status IN ('queued','running')", (session_id,)).fetchone():
                raise ValueError("这个会话还有任务执行中，请等待或停止后再发送。")
            db.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?)", (identity, session_id, "queued", json.dumps(context), "", now, now))
        return identity

    def task(self, identity):
        with self.db() as db:
            row = db.execute("SELECT * FROM tasks WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise ValueError("任务不存在")
        result = dict(row)
        result["context"] = json.loads(result["context"])
        return result

    def task_status(self, identity, status, error=""):
        with self.db() as db:
            db.execute("UPDATE tasks SET status=?,error=?,updated=? WHERE id=?", (status, error, time.time(), identity))

    def finish(self, task_id, status, kind, data, error="", assistant=None):
        """Commit terminal status, answer and SSE event in one transaction."""
        now = time.time()
        with self.db() as db:
            task = db.execute("SELECT session_id,status FROM tasks WHERE id=?", (task_id,)).fetchone()
            if task is None or task["status"] not in ("queued", "running"):
                return None
            db.execute("UPDATE tasks SET status=?,error=?,updated=? WHERE id=?", (status, error, now, task_id))
            if assistant is not None:
                db.execute("INSERT INTO messages(session_id,role,content,task_id,created) VALUES(?,?,?,?,?)",
                           (task["session_id"], "assistant", assistant, task_id, now))
                db.execute("UPDATE sessions SET updated=? WHERE id=?", (now, task["session_id"]))
            row = db.execute("INSERT INTO events(task_id,type,data,created) VALUES(?,?,?,?)",
                             (task_id, kind, json.dumps(data, ensure_ascii=False, allow_nan=False), now))
            return {"id": row.lastrowid, "task_id": task_id, "type": kind, "data": data, "created": now}

    def event(self, task_id, kind, data):
        now = time.time()
        with self.db() as db:
            row = db.execute("INSERT INTO events(task_id,type,data,created) VALUES(?,?,?,?)",
                             (task_id, kind, json.dumps(data, ensure_ascii=False, allow_nan=False), now))
            return {"id": row.lastrowid, "task_id": task_id, "type": kind, "data": data, "created": now}

    def events(self, task_id, after=0):
        with self.db() as db:
            rows = db.execute("SELECT * FROM events WHERE task_id=? AND id>? ORDER BY id", (task_id, after)).fetchall()
        return [{**dict(r), "data": json.loads(r["data"])} for r in rows]

    def session_tasks(self, session_id):
        with self.db() as db:
            return [dict(r) for r in db.execute("SELECT id,status,error,created FROM tasks WHERE session_id=? ORDER BY created", (session_id,))]

    def artifacts(self, session_id):
        with self.db() as db:
            rows = db.execute("SELECT e.data FROM events e JOIN tasks t ON t.id=e.task_id WHERE t.session_id=? AND e.type='artifact' ORDER BY e.id", (session_id,)).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def summary(self):
        with self.db() as db:
            counts = {r["status"]: r["n"] for r in db.execute("SELECT status,COUNT(*) n FROM tasks GROUP BY status")}
            sessions = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            recent = [dict(row) for row in db.execute("SELECT t.id,t.status,t.error,t.created,s.title FROM tasks t JOIN sessions s ON t.session_id=s.id ORDER BY t.created DESC LIMIT 6")]
        return {"sessions": sessions, "tasks": sum(counts.values()), "completed": counts.get("completed",0), "running": counts.get("running",0)+counts.get("queued",0), "failed": counts.get("failed",0), "recent": recent}
