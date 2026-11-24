from io import BytesIO
from unittest.mock import patch
import json
import unittest

from research_agent.diagnostics import check_environment
from research_agent.evidence import inspect


def source(text):
    return {"source_id": "notice", "name": "notice.pdf",
            "pages": [{"locator": "第 1 页", "text": text}]}


class NoticeTests(unittest.TestCase):
    def test_explicit_fields_have_unique_citations(self):
        material = source("成果类型：论文\n成果名称：示例题名\n状态：已录用\n证明类型：稿件录用通知")
        citations = inspect(material)["citations"]
        self.assertEqual(len(citations), len({item["quote"] for item in citations}))

    def test_unstructured_acceptance_notice(self):
        material = source("《示例期刊》稿件录用通知\n您的论文《示例论文题名》已收悉。\n经审查，同意采用，拟于明年刊出。")
        record = inspect(material)
        self.assertEqual(record["title"], "示例论文题名")
        self.assertEqual(record["status_code"], "paper_accepted")
        for citation in record["citations"]:
            self.assertIn(citation["quote"], material["pages"][0]["text"])

    def test_unstructured_patent_receipt(self):
        text = "专 利 申 请 受 理 通 知 书\n申请人提出的专利申请已由国家知识产权局受理。\n发明创造名称：示例训练装置\n发明人：作者甲，作者乙"
        record = inspect(source(text))
        self.assertEqual(record["title"], "示例训练装置")
        self.assertEqual(record["status_code"], "patent_received")

    def test_negative_acceptance_is_not_positive(self):
        text = "稿件录用通知\n论文《示例题名》\n终审意见为不同意采用。"
        self.assertEqual(inspect(source(text))["status_code"], "unknown")

    def test_header_without_acceptance_body_is_unknown(self):
        text = "稿件录用通知\n论文《示例题名》\n稿件正在审查。"
        self.assertEqual(inspect(source(text))["status_code"], "unknown")

    def test_positive_and_negative_notice_is_conflict(self):
        text = "稿件录用通知\n论文《示例题名》\n初审同意采用。\n终审不同意采用。"
        self.assertEqual(inspect(source(text))["status_code"], "conflict")

    def test_future_issue_alone_does_not_mean_published(self):
        record = inspect(source("《期刊》稿件录用通知\n论文《示例题名》\n同意录用，计划 2027 年第 1 期发表。"))
        self.assertEqual(record["status_code"], "paper_accepted")


class DiagnosticTests(unittest.TestCase):
    def test_installed_model(self):
        with patch("research_agent.diagnostics.urlopen", return_value=BytesIO(
                json.dumps({"models": [{"name": "qwen3:latest"}]}).encode())):
            result = check_environment("http://127.0.0.1:11434", "qwen3")
        self.assertTrue(result["model_available"])

    def test_unavailable_model(self):
        with patch("research_agent.diagnostics.urlopen", return_value=BytesIO(b'{"models":[]}')):
            result = check_environment("http://127.0.0.1:11434", "qwen3")
        self.assertTrue(result["service_reachable"])
        self.assertFalse(result["model_available"])


if __name__ == "__main__":
    unittest.main()
