"""Transactional document ingestion and BM25 retrieval."""

from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import math
import re
import sqlite3


def tokens(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.extend(run[i:i + 2] for i in range(len(run) - 1))
        if len(run) == 1:
            words.append(run)
    return words


def read_pages(path):
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return [path.read_text(encoding="utf-8-sig")]
    if suffix == ".pdf":
        from pypdf import PdfReader
        pages = [page.extract_text() or "" for page in PdfReader(path).pages]
        if any(not page.strip() for page in pages):
            raise ValueError(f"{path.name} 含无文本页面，请先 OCR 后导入")
        return pages
    if suffix == ".docx":
        from docx import Document
        document = Document(path)
        # DOCX has no stable page positions before rendering. Use a document locator.
        namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        return ["\n".join(
            "".join(t.text or "" for t in paragraph.iter(namespace + "t"))
            for paragraph in document.element.body.iter(namespace + "p")
        )]
    raise ValueError(f"不支持的文件类型 {suffix}")


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL,
                digest TEXT NOT NULL, pages TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
                page INTEGER NOT NULL, text TEXT NOT NULL,
                FOREIGN KEY(source_id) REFERENCES sources(id) ON DELETE CASCADE
            );
        """)

    def close(self):
        self.connection.close()

    def ingest(self, directory):
        root = Path(directory).resolve()
        if not root.is_dir():
            raise ValueError("材料目录不存在")
        paths = sorted(path for path in root.rglob("*") if path.is_file()
                       and path.suffix.lower() in (".pdf", ".docx", ".txt", ".md")
                       and not path.name.startswith("~$"))
        if not paths:
            raise ValueError("目录中没有支持的材料")
        prepared = []
        for path in paths:
            pages = read_pages(path)
            if not "".join(pages).strip():
                raise ValueError(f"{path.name} 为空")
            name = path.relative_to(root).as_posix()
            source_id = "D" + sha256(name.encode()).hexdigest()[:12]
            prepared.append((source_id, name, sha256(path.read_bytes()).hexdigest(), pages))
        # Parse all inputs first. A parser failure keeps the prior collection intact.
        with self.connection:
            self.connection.execute("DELETE FROM sources")
            for source_id, name, digest, pages in prepared:
                self.connection.execute("INSERT INTO sources VALUES (?, ?, ?, ?)",
                                        (source_id, name, digest, json.dumps(pages, ensure_ascii=False)))
                for page_number, page in enumerate(pages, 1):
                    for index, start in enumerate(range(0, len(page), 520)):
                        self.connection.execute("INSERT INTO chunks VALUES (?, ?, ?, ?)",
                                                (f"{source_id}-{page_number}-{index}", source_id,
                                                 page_number, page[start:start + 650]))
        return self.list_sources()

    def list_sources(self):
        return [dict(zip(("source_id", "name", "sha256"), row)) for row in
                self.connection.execute("SELECT id, name, digest FROM sources ORDER BY name")]

    def read_source(self, source_id):
        row = self.connection.execute("SELECT name, pages FROM sources WHERE id = ?",
                                      (source_id,)).fetchone()
        if row is None:
            raise ValueError("source_id 不存在")
        pages = json.loads(row[1])
        if sum(map(len, pages)) > 30000:
            raise ValueError("材料超过 30000 字，请按成果拆分后导入")
        is_pdf = row[0].lower().endswith(".pdf")
        return {"source_id": source_id, "name": row[0], "pages": [
            {"locator": f"第 {i} 页" if is_pdf else "全文", "text": text}
            for i, text in enumerate(pages, 1)
        ]}

    def search(self, query, limit=5):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须为非空字符串")
        if type(limit) is not int or not 1 <= limit <= 10:
            raise ValueError("limit 应为 1 至 10 的整数")
        rows = self.connection.execute("""
            SELECT c.source_id, s.name, c.page, c.text FROM chunks c
            JOIN sources s ON c.source_id = s.id ORDER BY c.id
        """).fetchall()
        counts = [Counter(tokens(row[3])) for row in rows]
        lengths = [sum(count.values()) for count in counts]
        average = sum(lengths) / max(1, len(rows))
        terms = set(tokens(query))
        frequency = {term: sum(term in count for count in counts) for term in terms}
        results = []
        for row, count, length in zip(rows, counts, lengths):
            score = 0.0
            for term in terms:
                occurrences = count[term]
                if occurrences:
                    idf = math.log(1 + (len(rows) - frequency[term] + .5) / (frequency[term] + .5))
                    score += idf * occurrences * 2.5 / (occurrences + 1.5 * (.25 + .75 * length / average))
            if score > 0:
                results.append({"source_id": row[0], "name": row[1], "page": row[2],
                                "text": row[3], "score": round(score, 4)})
        return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]
