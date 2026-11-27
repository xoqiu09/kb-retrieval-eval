from math import log
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from kbeval.__main__ import load_data, make_chunks, query_result
from kbeval.chunk import Chunk, by_heading, fixed_width, gold_chunks
from kbeval.metrics import summarize
from kbeval.retrieve import BM25


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_TEXT = re.compile(
    r"(?<![A-Za-z0-9_./\\~-])(?:/+|~[/\\]|[A-Za-z]:[/\\]|\\\\)[^\s/\\]"
    r"|[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}"
    r"|(?<![0-9])1[3-9][0-9]{9}(?![0-9])"
    r"|-----BEGIN .*PRIVATE KEY-----"
)


def json_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        yield from value
        for field in value.values():
            yield from json_strings(field)
    elif isinstance(value, list):
        for item in value:
            yield from json_strings(item)


def cli(*args: str) -> str:
    return subprocess.run(
        [sys.executable, "-m", "kbeval", *args], cwd=ROOT,
        check=True, capture_output=True, encoding="utf-8",
    ).stdout


class ChunkTests(unittest.TestCase):
    def test_heading_paths_and_source_ranges(self):
        text = "导语\n# 售后\n## 退货 ##\n### 运费\n正文\n## 保修\n# 安装\n正文"
        chunks = by_heading("manual.md", text)
        self.assertEqual([(chunk.title_path, chunk.text) for chunk in chunks], [
            ((), "导语\n"),
            (("售后",), "# 售后\n"),
            (("售后", "退货"), "## 退货 ##\n"),
            (("售后", "退货", "运费"), "### 运费\n正文\n"),
            (("售后", "保修"), "## 保修\n"),
            (("安装",), "# 安装\n正文"),
        ])
        for chunk in chunks:
            self.assertEqual(chunk.document, "manual.md")
            self.assertEqual(chunk.text, text[chunk.start:chunk.end])

    def test_fixed_windows_keep_exact_overlap_and_tail(self):
        text = "# 总览\n" + "甲" * 135 + "\n## 细则\n" + "乙" * 325 + "终"
        for overlap in (0, 50):
            with self.subTest(overlap=overlap):
                chunks = fixed_width("manual.md", text, 200, overlap)
                self.assertEqual(chunks[0].start, 0)
                self.assertEqual(chunks[0].title_path, ("总览",))
                self.assertEqual(chunks[1].title_path, ("总览", "细则"))
                self.assertEqual(chunks[-1].end, len(text))
                self.assertTrue(chunks[-1].text.endswith("终"))
                for chunk in chunks:
                    self.assertEqual(chunk.text, text[chunk.start:chunk.end])
                    self.assertLessEqual(len(chunk.text), 200)
                for left, right in zip(chunks, chunks[1:]):
                    self.assertEqual(left.end - right.start, overlap)
                    self.assertEqual(left.text[len(left.text) - overlap:], right.text[:overlap])

    def test_q07_overlap_recovers_a_split_gold_sentence(self):
        documents, questions = load_data()
        question = next(question for question in questions if question["id"] == "Q07")
        gold = question["gold"]
        start = documents["delivery.md"].index(gold)
        self.assertLess(start, 200)
        self.assertGreater(start + len(gold), 200)

        no_overlap = make_chunks(documents, "fixed-200-0")
        result = query_result(question, BM25(no_overlap).rank(question["question"]), gold_chunks(no_overlap, gold), 3)
        self.assertEqual(result["gold_chunks"], [])
        self.assertEqual(result["diagnosis"]["reason"], "答案被切断")

        with_overlap = make_chunks(documents, "fixed-200-50")
        result = query_result(question, BM25(with_overlap).rank(question["question"]), gold_chunks(with_overlap, gold), 3)
        self.assertEqual(result["gold_chunks"], ["delivery.md:150-350"])
        self.assertIsNone(result["diagnosis"])


class ScoringTests(unittest.TestCase):
    def test_bm25_matches_three_document_hand_calculation(self):
        texts = {"a.md": "猫 猫 桌 椅 书 灯 路 门 箱 床", "b.md": "猫 狗", "c.md": "鱼 鸟"}
        chunks = [Chunk(name, (), 0, len(text), text) for name, text in texts.items()]
        ranked = BM25(chunks).rank("猫")
        # Token lengths 10, 2, 2; average 14/3; df(猫)=2, IDF=ln(1.6).
        # With k1=1.2 and b=0.75, the tf factors for a and b are 77/74 and 77/59.
        self.assertEqual([chunk.document for chunk, _ in ranked], ["b.md", "a.md", "c.md"])
        for (_, score), expected in zip(ranked, [log(1.6) * 77 / 59, log(1.6) * 77 / 74, 0]):
            self.assertAlmostEqual(score, expected)

    def test_recall_and_mrr_match_hand_calculation(self):
        ranks = {"first": 1, "third": 3, "fifth": 5, "missing": None}
        mrr = (1 + 1 / 3 + 1 / 5) / 4
        for k, recall, late in (
            (1, 1 / 4, {}),
            (3, 2 / 4, {"third": 3}),
            (5, 3 / 4, {"third": 3, "fifth": 5}),
        ):
            with self.subTest(k=k):
                result = summarize(ranks, k)
                self.assertAlmostEqual(result["recall"], recall)
                self.assertAlmostEqual(result["mrr"], mrr)
                self.assertEqual(result["late_hits"], late)


class CommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.directory = Path(tmp.name)
        cls.baseline = cls.directory / "report.json"
        cli("eval", "--out", str(cls.baseline))
        cls.report = json.loads(cls.baseline.read_bytes())

    def test_eval_matrix_and_gold_are_independent_of_retriever(self):
        documents, questions = load_data()
        self.assertEqual(len(documents), 6)
        self.assertEqual(len(questions), 40)
        self.assertEqual(self.report["questions"], questions)
        chunkings = ("heading", "fixed-200-0", "fixed-200-50")
        configs = self.report["configs"]
        self.assertEqual(set(configs), {
            f"{chunking}/{retriever}/{k}" for chunking in chunkings
            for retriever in ("bm25", "bigram") for k in (1, 3, 5)
        })
        for chunking in chunkings:
            chunks = self.report["chunks"][chunking]
            chunk_text = {cid: documents[c["document"]][c["start"]:c["end"]] for cid, c in chunks.items()}
            for retriever in ("bm25", "bigram"):
                for k in (1, 3, 5):
                    row = configs[f"{chunking}/{retriever}/{k}"]
                    self.assertEqual(row["chunk_count"], len(chunks))
                    self.assertEqual(set(row["results"]), {item["id"] for item in questions})
                    for item in questions:
                        expected = sorted(cid for cid, text in chunk_text.items() if item["gold"] in text)
                        self.assertEqual(row["results"][item["id"]]["gold_chunks"], expected)

        self.assertEqual(configs["heading/bm25/3"]["results"]["Q03"]["rank"], 3)
        self.assertEqual(configs["heading/bigram/3"]["results"]["Q03"]["rank"], 1)

    def test_eval_uses_the_requested_recall_cutoff(self):
        configs = self.report["configs"]
        self.assertEqual(configs["heading/bm25/1"]["recall"], 37 / 40)
        self.assertEqual(configs["heading/bm25/3"]["recall"], 39 / 40)

    def test_recommendation_prefers_mrr_when_recall_ties(self):
        configs = self.report["configs"]
        heading = configs["heading/bigram/3"]
        fixed = configs["fixed-200-50/bm25/3"]
        self.assertEqual(heading["recall"], fixed["recall"])
        self.assertGreater(heading["mrr"], fixed["mrr"])
        self.assertGreater(heading["chunk_count"], fixed["chunk_count"])
        self.assertEqual(list(self.report)[-1], "recommendation")
        self.assertEqual(self.report["recommendation"]["config"], "heading/bigram/3")
        self.assertIn("仅适用于这组虚构数据", self.report["recommendation"]["reason"])

    def test_eval_is_byte_identical_after_rerun(self):
        second = self.directory / "report-again.json"
        cli("eval", "--out", str(second))
        self.assertEqual(self.baseline.read_bytes(), second.read_bytes())

    def test_compare_reports_changes_and_counts(self):
        left = self.directory / "before.json"
        right = self.directory / "after.json"
        old = {"Q01": None, "Q02": 1, "Q03": 2, "Q04": 4, "Q05": 3}
        new = {"Q01": 2, "Q02": 3, "Q03": 2, "Q04": 6, "Q05": 2}
        questions = self.report["questions"][:5]
        for path, ranks, records in ((left, old, questions), (right, new, questions[::-1])):
            report = {
                "questions": records,
                "configs": {
                    f"heading/bm25/{k}": {"results": {
                        qid: {"rank": rank} for qid, rank in ranks.items()
                    }} for k in (1, 3)
                },
            }
            path.write_text(json.dumps(report), encoding="utf-8")
        self.assertEqual(cli("compare", str(left), str(right)).splitlines(), [
            "heading/bm25/1｜变好 0 题：无｜变坏 1 题：Q02｜不变 4 题",
            "heading/bm25/3｜变好 2 题：Q01, Q05｜变坏 1 题：Q02｜不变 2 题",
            "2 格，共 10 项逐题对比：变好 2，变坏 2，不变 6。",
        ])

    def test_compare_rejects_changed_question_or_gold(self):
        right = self.directory / "changed-question.json"
        for field in ("question", "gold"):
            with self.subTest(field=field):
                report = json.loads(self.baseline.read_bytes())
                questions = {item["id"]: item for item in report["questions"]}
                questions["Q07"][field] = questions["Q34"][field]
                right.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    cli("compare", str(self.baseline), str(right))
                self.assertEqual(caught.exception.returncode, 2)
                self.assertEqual(caught.exception.stdout, "")
                self.assertIn("Q07", caught.exception.stderr)

    def test_explain_prints_each_diagnosis_with_evidence(self):
        cut = cli("explain", "Q07", "--config", "fixed-200-0/bm25/3")
        self.assertIn("归因：答案被切断", cut)
        self.assertIn("1. delivery.md", cut)
        self.assertIn("命中前 3 名", cli("explain", "Q07", "--config", "fixed-200-50/bm25/3"))

        no_overlap = cli("explain", "Q34", "--config", "heading/bm25/3")
        self.assertIn("归因：关键词不重合", no_overlap)
        detail = self.report["configs"]["heading/bm25/3"]["results"]["Q34"]["diagnosis"]
        lines = no_overlap.splitlines()
        self.assertIn(f"问题词：{', '.join(detail['question_terms'])}", lines)
        self.assertIn(f"原句词：{', '.join(detail['gold_terms'])}", lines)
        self.assertTrue({"遥测", "档案"} <= set(detail["question_terms"]))
        self.assertTrue({"云端", "日志"} <= set(detail["gold_terms"]))
        self.assertTrue(set(detail["question_terms"]).isdisjoint(detail["gold_terms"]))

        displaced = cli("explain", "Q03", "--config", "heading/bm25/1")
        self.assertIn("归因：被相似段落挤掉", displaced)
        row = self.report["configs"]["heading/bm25/1"]["results"]["Q03"]
        ahead = row["diagnosis"]["ahead"]
        self.assertEqual(len(ahead), row["rank"] - 1)
        self.assertEqual(ahead[0], row["top"][0]["chunk"])
        self.assertIn(f"排在首个含原句块之前的块：{', '.join(ahead)}", displaced)

    def test_export_is_stable_and_uses_only_public_corpus_sources(self):
        first, second = (self.directory / name for name in ("demo.json", "demo-again.json"))
        cli("export", "--out", str(first))
        cli("export", "--out", str(second))
        self.assertEqual(first.read_bytes(), second.read_bytes())
        export = json.loads(first.read_bytes())
        self.assert_export_has_no_private_text(export)

        documents, _ = load_data()
        self.assertEqual(set(export), {"questions", "chunks", "configs", "recommendation"})
        for chunks in export["chunks"].values():
            for cid, chunk in chunks.items():
                self.assertEqual(set(chunk), {"document", "title_path", "start", "end", "text"})
                name, start, end = chunk["document"], chunk["start"], chunk["end"]
                self.assertEqual(cid, f"{name}:{start}-{end}")
                self.assertEqual(chunk["text"], documents[name][start:end])
                del chunk["text"]
        self.assertEqual(export, self.report)

    def test_export_privacy_check_rejects_paths_and_phones_in_prose(self):
        for text in ("详见 `/tmp/demo.md`。", "请联系13900000000确认。"):
            with self.subTest(text=text):
                payload = json.loads(json.dumps({"nested": [{"detail": text}]}))
                with self.assertRaisesRegex(AssertionError, "发现路径或隐私信息"):
                    self.assert_export_has_no_private_text(payload)

    def assert_export_has_no_private_text(self, export):
        for value in json_strings(export):
            self.assertNotIn(str(ROOT), value)
            self.assertNotIn(str(self.directory), value)
            self.assertNotRegex(value, PRIVATE_TEXT, "发现路径或隐私信息")
