from math import fsum, log
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from kbeval.chunk import Chunk
from kbeval.retrieve import BM25, BigramOverlap, expand_query


ROOT = Path(__file__).resolve().parents[1]


def cli(*args: str) -> str:
    return subprocess.run(
        [sys.executable, "-m", "kbeval", *args], cwd=ROOT,
        check=True, capture_output=True, encoding="utf-8",
    ).stdout


class ExpansionScoringTests(unittest.TestCase):
    def test_expansion_matches_original_query_once(self):
        synonyms = {"Wi-Fi": ["无线网络"], "无线网络": ["配网"]}
        self.assertEqual(expand_query("支持 WI-FI 吗？", synonyms), "无线网络")
        self.assertEqual(expand_query("保修多久？", synonyms), "")

    def test_expansion_applies_hand_calculated_weights(self):
        texts = {"a.md": "无线", "b.md": "联网", "c.md": "线网"}
        chunks = [Chunk(name, (), 0, len(text), text) for name, text in texts.items()]
        expansion = expand_query("无线", {"无线": ["无线", "联网", "联网"]})

        # Each chunk has three BM25 tokens with tf=1, so the length and tf factors cancel.
        # 线 and 网 each occur in two chunks. All other terms occur in one chunk.
        rare, shared = log(8 / 3), log(8 / 5)
        for weight, options in ((0.5, {}), (0.25, {"expansion_weight": 0.25})):
            for retriever, expected in (
                (BM25, {"a.md": 2 * rare + shared, "b.md": weight * (2 * rare + shared),
                        "c.md": (1 + weight) * shared}),
                (BigramOverlap, {"a.md": 1, "b.md": weight, "c.md": 0}),
            ):
                with self.subTest(retriever=retriever.__name__, weight=weight):
                    ranked = retriever(chunks).rank("无线", expansion=expansion, **options)
                    self.assertEqual(
                        [chunk.document for chunk, _ in ranked],
                        sorted(expected, key=lambda name: (-expected[name], name)),
                    )
                    for chunk, score in ranked:
                        self.assertAlmostEqual(score, expected[chunk.document])


class ExpansionCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.path = cls.directory / "expanded.json"
        cls.eval_output = cli("eval", "--expand", "--out", str(cls.path))
        cls.report = json.loads(cls.path.read_bytes())

    def test_expansion_off_reproduces_all_m1_configs(self):
        path = self.directory / "m1.json"
        cli("eval", "--out", str(path))
        baseline = json.loads(path.read_bytes())
        expected_non_first = {
            "heading/bm25": {"Q03": 3, "Q34": 16, "Q39": 2},
            "heading/bigram": {"Q34": 12},
            "fixed-200-0/bm25": {"Q06": None, "Q07": None, "Q10": None, "Q26": None, "Q34": 11},
            "fixed-200-0/bigram": {"Q06": None, "Q07": None, "Q10": None, "Q26": None, "Q34": 7},
            "fixed-200-50/bm25": {"Q34": 15},
            "fixed-200-50/bigram": {"Q05": 2, "Q34": 7},
        }
        self.assertEqual(len(baseline["configs"]), 18)
        self.assertEqual(len(self.report["configs"]), 36)
        self.assertEqual(self.report["questions"], baseline["questions"])
        for name, row in baseline["configs"].items():
            with self.subTest(config=name):
                settings, cutoff = name.rsplit("/", 1)
                k = int(cutoff)
                ranks = {item["id"]: expected_non_first[settings].get(item["id"], 1)
                         for item in baseline["questions"]}
                self.assertEqual({qid: result["rank"] for qid, result in row["results"].items()}, ranks)
                self.assertEqual(row["recall"], sum(1 for rank in ranks.values() if rank and rank <= k) / 40)
                self.assertAlmostEqual(row["mrr"], fsum(1 / rank for rank in ranks.values() if rank) / 40)
                self.assertEqual(self.report["configs"][name], row)
                enabled = self.report["configs"][f"{name}/expanded"]
                self.assertEqual(
                    {qid: result["gold_chunks"] for qid, result in enabled["results"].items()},
                    {qid: result["gold_chunks"] for qid, result in row["results"].items()},
                )

    def test_compare_lists_improvement_and_the_chunk_that_displaced_gold(self):
        output = cli("compare", str(self.path)).splitlines()
        row = "heading/bigram/1｜变好 1 题：Q34｜变坏 1 题：Q39｜不变 38 题"
        self.assertIn(row, output)
        self.assertEqual(
            output[output.index(row) + 1],
            "  Q39｜第 1 名 → 第 2 名｜新超过标准答案块：smart.md:292-378",
        )
        self.assertIn("18 格，共 720 项逐题对比：变好 23，变坏 3，不变 694。", output)

    def test_compare_expansion_rejects_a_question_missing_from_both_configs(self):
        config = "heading/bigram/1"
        path = self.directory / "missing-question.json"
        for qid in ("Q39", "H01"):
            with self.subTest(qid=qid):
                report = json.loads(self.path.read_bytes())
                dataset = report if qid == "Q39" else report["held_out"]
                for name in (config, f"{config}/expanded"):
                    del dataset["configs"][name]["results"][qid]
                path.write_text(json.dumps(report), encoding="utf-8")

                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    cli("compare", str(path))
                self.assertEqual(caught.exception.returncode, 2)
                self.assertEqual(caught.exception.stdout, "")
                self.assertIn(config, caught.exception.stderr)
                self.assertIn(qid, caught.exception.stderr)

    def test_compare_two_reports_rejects_missing_held_out_group(self):
        path = self.directory / "missing-held-out.json"
        report = json.loads(self.path.read_bytes())
        del report["held_out"]
        path.write_text(json.dumps(report), encoding="utf-8")
        for left, right, missing in ((path, self.path, "第一份"), (self.path, path, "第二份")):
            with self.subTest(missing=missing):
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    cli("compare", str(left), str(right))
                self.assertEqual(caught.exception.returncode, 2)
                self.assertEqual(caught.exception.stdout, "")
                self.assertIn(f"{missing}报告缺少留出题", caught.exception.stderr)

    def test_held_out_is_separate_in_eval_and_compare(self):
        held = self.report["held_out"]
        self.assertIn("36 格、原有 40 题及留出 8 题", self.eval_output)
        self.assertEqual(set(held["configs"]), set(self.report["configs"]))
        held_ids = {f"H{i:02}" for i in range(1, 9)}
        self.assertEqual({item["id"] for item in held["questions"]}, held_ids)
        self.assertTrue({item["question"] for item in held["questions"]}.isdisjoint(
            item["question"] for item in self.report["questions"]
        ))
        for name, row in self.report["configs"].items():
            self.assertEqual(set(row["results"]), {f"Q{i:02}" for i in range(1, 41)})
            self.assertEqual(set(held["configs"][name]["results"]), held_ids)
        for name in ("heading/bm25/3", "heading/bm25/3/expanded"):
            self.assertEqual(held["configs"][name]["recall"], 7 / 8)
            self.assertAlmostEqual(held["configs"][name]["mrr"], 0.90625)
        for name, score in (("heading/bigram/1", 2), ("heading/bigram/1/expanded", 2.5)):
            h01 = held["configs"][name]["results"]["H01"]
            self.assertEqual(h01["rank"], 1)
            self.assertEqual(h01["top"][0], {"chunk": "delivery.md:9-189", "score": score})
        self.assertEqual(cli("compare", str(self.path)).splitlines()[-2:], [
            "留出题（8 题）",
            "18 格，共 144 项逐题对比：变好 0，变坏 0，不变 144。",
        ])

    def test_eval_uses_the_configured_expansion_weight(self):
        path = self.directory / "weighted.json"
        cli("eval", "--expand", "--expansion-weight", "0.4", "--out", str(path))
        report = json.loads(path.read_bytes())
        self.assertEqual(self.report["expansion_weight"], 0.5)
        self.assertEqual(report["expansion_weight"], 0.4)
        q34 = report["configs"]["heading/bigram/1/expanded"]["results"]["Q34"]
        self.assertEqual(q34["top"][0]["chunk"], "account.md:290-374")
        self.assertAlmostEqual(q34["top"][0]["score"], 1.2)
