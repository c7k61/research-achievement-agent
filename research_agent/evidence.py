"""Conservative extraction for explicitly labeled achievement notices."""

import re


STATUSES = {
    "paper_accepted": "论文已录用",
    "paper_published": "论文已发表",
    "patent_received": "专利申请已受理",
    "patent_granted": "专利已授权",
    "conflict": "材料含相互冲突的状态，需人工核对",
    "unknown": "材料不足，需人工核对",
}


def add_notice_fields(fields, lines):
    """Recognize two explicit Chinese notice templates and retain original quotes."""
    header_lines = [item for item in lines if not re.match(r"^证明类型\s*[:：]", item[1])]
    paper_header = next((item for item in header_lines if "稿件录用通知" in item[1]), None)
    patent_header = next((item for item in header_lines if "专利申请受理通知书" in re.sub(r"\s", "", item[1])), None)
    if not paper_header and not patent_header:
        return

    def add(label, evidence, value):
        locator, quote = evidence
        fields.setdefault(label, []).append((locator, quote, value))

    if paper_header:
        add("成果类型", paper_header, "论文")
        add("证明类型", paper_header, "稿件录用通知")
        for locator, line in lines:
            title = re.search(r"论文\s*《([^》]+)》", line)
            if title:
                add("成果名称", (locator, line), title[1])
            if re.search(r"(?<!不)(?:同意采用|同意录用|予以录用)", line):
                add("状态", (locator, line), "已录用")
            if any(term in line for term in ("不同意采用", "不同意录用", "不予录用", "拒稿")):
                add("状态", (locator, line), "未录用")
        journal = re.search(r"《([^》]+)》", paper_header[1])
        if journal:
            add("期刊", paper_header, journal[1])

    if patent_header:
        add("成果类型", patent_header, "专利")
        add("证明类型", patent_header, "专利申请受理通知书")
        for locator, line in lines:
            title = re.match(r"^发明创造名称\s*[:：]\s*(.+)$", line)
            if title:
                add("成果名称", (locator, line), title[1])
            compact = re.sub(r"\s", "", line)
            if "专利申请已由国家知识产权局受理" in compact or "专利申请已受理" in compact:
                add("状态", (locator, line), "已受理")
            if "专利申请未受理" in compact or "专利申请不予受理" in compact:
                add("状态", (locator, line), "未受理")


def inspect(source):
    lines = [(page["locator"], line.strip()) for page in source["pages"]
             for line in page["text"].splitlines() if line.strip()]
    fields = {}
    for locator, line in lines:
        match = re.match(r"^(成果类型|成果名称|作者|发明人|期刊|申请号|状态|证明类型)\s*[:：]\s*(.+)$", line)
        if match:
            fields.setdefault(match[1], []).append((locator, line, match[2].strip()))
    add_notice_fields(fields, lines)

    def value(label):
        candidates = fields.get(label, [])
        unique = list(dict.fromkeys(item[2] for item in candidates))
        return unique[0] if len(unique) == 1 else ""

    kind = value("成果类型")
    status_lines = fields.get("状态", [])
    declared = {item[2] for item in status_lines}
    states = {"论文": {"已录用": "paper_accepted", "已发表": "paper_published"},
              "专利": {"已受理": "patent_received", "已授权": "patent_granted"}}
    code = states.get(kind, {}).get(value("状态"), "unknown")
    proof_types = {"paper_accepted": "稿件录用通知", "paper_published": "期刊正式发表记录",
                   "patent_received": "专利申请受理通知书", "patent_granted": "发明专利证书"}
    if proof_types.get(code) != value("证明类型"):
        code = "unknown"
    if len(declared) > 1:
        code = "conflict"
    title = value("成果名称")
    if not title:
        code = "unknown"
    citations = [{"source_id": source["source_id"], "name": source["name"],
                  "locator": locator, "quote": line}
                 for label in ("成果名称", "状态", "证明类型")
                 for locator, line, _ in fields.get(label, [])]
    # All text in the display comes from validated fields or controlled status labels.
    return {"source_id": source["source_id"], "title": title or source["name"],
            "status_code": code, "status": STATUSES[code],
            "contributors": value("作者") or value("发明人"),
            "reference": value("期刊") or value("申请号"), "citations": citations}


def format_answer(question, records):
    lines = ["核验问题", question, "", "核验结果"]
    if not records:
        return "材料不足，未生成成果表述。"
    for record in records:
        lines.extend([f"• {record['title']}　{record['status']}",
                      f"  署名 {record['contributors'] or '待核对'}　{record['reference']}"])
        for citation in record["citations"]:
            lines.append(f"  [{citation['source_id']} · {citation['name']} · {citation['locator']}] {citation['quote']}")
    lines.extend(["", "状态依据为导入材料中的明确字段。材料真实性和署名贡献由使用者核对。"])
    return "\n".join(lines)
