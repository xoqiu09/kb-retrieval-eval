from io import StringIO
from math import fsum, log
import json
from pathlib import Path
import re
import shutil
from shutil import copytree
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kbeval.__main__ import CONFIGS, CUTOFFS, EXPANDED_SUFFIX, build_report, main
from kbeval.__main__ import main
from kbeval.chunk import Chunk
from kbeval.metrics import reciprocal_rank
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

        synonyms = {"E02": ["水箱没有装到位"], "refund": ["申请退款"]}
        for question, expected in (
            ("故障码e02？", "水箱没有装到位"),
            ("故障码 E020？", ""),
            ("故障码 AE02？", ""),
            ("Refunds?", "申请退款"),
        ):
            with self.subTest(question=question):
                self.assertEqual(expand_query(question, synonyms), expected)

        documents = {"tank.md": "E02 表示水箱没有装到位。", "valve.md": "E020 表示阀门未就位。"}
        questions = [{"id": "Q01", "question": "E020？", "gold": documents["valve.md"]}]
        report = build_report(documents, questions, include_text=False, synonyms=synonyms)
        for retriever in ("bm25", "bigram"):
            for suffix in ("", EXPANDED_SUFFIX):
                config = f"heading/{retriever}/1{suffix}"
                with self.subTest(config=config):
                    self.assertEqual(report["configs"][config]["results"]["Q01"]["rank"], 1)

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

    def test_explain_accepts_the_recommended_expanded_config(self):
        config = self.report["recommendation"]["config"]
        self.assertTrue(config.endswith(EXPANDED_SUFFIX))
        result = self.report["configs"][config]["results"]["Q34"]
        self.assertEqual(result["rank"], 1)
        hit = result["top"][0]
        self.assertIn(hit["chunk"], result["gold_chunks"])
        chunking = config.split("/", 1)[0]
        source = self.report["chunks"][chunking][hit["chunk"]]

        explained = subprocess.run(
            [sys.executable, "-m", "kbeval", "explain", "Q34", "--config", config],
            cwd=ROOT, capture_output=True, encoding="utf-8",
        )
        self.assertEqual(explained.returncode, 0, explained.stderr)
        self.assertIn("首个含标准原句的块排第 1 名。", explained.stdout)
        title = " / ".join(source["title_path"])
        self.assertIn(
            f"1. {source['document']} [{source['start']}, {source['end']})｜{title}"
            f"｜得分 {hit['score']:.4f}｜含标准原句",
            explained.stdout.splitlines(),
        )

    def test_explain_reproduces_the_reports_custom_expansion_weight(self):
        path = self.directory / "custom-weight.json"
        config = "fixed-200-50/bm25/3/expanded"
        for weight, rank in ((0.2, 6), (0, 15)):
            with self.subTest(weight=weight):
                cli("eval", "--expand", "--expansion-weight", str(weight), "--out", str(path))
                report = json.loads(path.read_bytes())
                self.assertEqual(report["expansion_weight"], weight)
                if weight:
                    self.assertEqual(report["recommendation"]["config"], config)
                result = report["configs"][config]["results"]["Q34"]
                self.assertEqual(result["rank"], rank)

                output = cli("explain", "Q34", "--config", config,
                             "--expansion-weight", str(report["expansion_weight"]))
                self.assertIn(f"首个含标准原句的块排第 {rank} 名。", output)
                self.assertIn(f"归因：{result['diagnosis']['reason']}", output)
                self.assertIn(f"得分 {result['top'][0]['score']:.4f}", output)

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

    def test_compare_lists_all_displacing_chunks_at_cutoff_one(self):
        config = "heading/bigram/1"
        gold = "归还物件。"
        complete = "  Q01｜第 1 名 → 第 3 名｜新超过标准答案块：a.md:0-7, b.md:0-5"
        incomplete = (
            "  Q01｜第 1 名 → 第 7 名｜新超过标准答案块：a.md:0-7, b.md:0-5, d.md:0-5, e.md:0-5, f.md:0-5"
            "（清单不完整，请用 eval --expand 重新生成报告）"
        )
        for question, legacy, extra_documents, expected in (
            ("归还", False, "", complete),
            ("咨询", False, "", complete),
            ("咨询", True, "", complete),
            ("咨询", True, "defg", incomplete),
        ):
            with self.subTest(question=question, legacy=legacy, extra_documents=extra_documents):
                answer_document = gold if question == "归还" else f"{gold}咨询。"
                documents = {"a.md": "配送预约签收。", "b.md": "配送预约。", "c.md": answer_document}
                documents.update({f"{name}.md": documents["b.md"] for name in extra_documents})
                report = build_report(
                    documents, [{"id": "Q01", "question": question, "gold": gold}],
                    include_text=False, synonyms={question: ["配送预约签收"]},
                    held_out=[{"id": "H01", "question": "签收", "gold": documents["a.md"]}],
                )
                configs = report["configs"]
                if legacy:
                    # Older reports omitted "ahead" for the keyword-mismatch diagnosis.
                    for cutoff in CUTOFFS:
                        diagnosis = configs[f"heading/bigram/{cutoff}/expanded"]["results"]["Q01"]["diagnosis"]
                        if diagnosis is not None:
                            self.assertEqual(diagnosis["reason"], "关键词不重合")
                            diagnosis.pop("ahead", None)
                result = configs[f"{config}/expanded"]["results"]["Q01"]
                self.assertEqual(len(result["top"]), 1)
                path = self.directory / "displaced.json"
                path.write_text(json.dumps(report), encoding="utf-8")

                output = cli("compare", str(path)).splitlines()
                row = f"{config}｜变好 0 题：无｜变坏 1 题：Q01｜不变 0 题"
                self.assertEqual(output[output.index(row) + 1], expected)

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

    def test_eval_rejects_an_original_question_in_held_out_data(self):
        data = self.directory / "duplicate-data"
        shutil.copytree(ROOT / "data", data)
        questions = (data / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        original = next(item for item in map(json.loads, questions) if item["id"] == "Q34")
        held_out = data / "held_out.jsonl"
        records = [json.loads(line) for line in held_out.read_text(encoding="utf-8").splitlines()]
        held_out_gold = next(item["gold"] for item in records if item["id"] == "H08")
        self.assertNotEqual(original["gold"], held_out_gold)
        output = self.directory / "duplicate-report.json"

        for gold in (original["gold"], held_out_gold):
            with self.subTest(same_gold=gold == original["gold"]):
                duplicate = dict(original, id="H08", gold=gold)
                updated = [duplicate if item["id"] == "H08" else item for item in records]
                held_out.write_text("\n".join(map(json.dumps, updated)) + "\n", encoding="utf-8")
                with (
                    patch("kbeval.__main__.DATA", data),
                    patch("kbeval.__main__.build_report", wraps=build_report) as score,
                    patch("sys.argv", ["kbeval", "eval", "--expand", "--out", str(output)]),
                    patch("sys.stdout", new_callable=StringIO),
                    patch("sys.stderr", new_callable=StringIO) as stderr,
                ):
                    with self.assertRaises(SystemExit) as caught:
                        main()
                score.assert_not_called()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("Q34", stderr.getvalue())
                self.assertIn("H08", stderr.getvalue())
                self.assertFalse(output.exists())

    def test_readme_matches_test_count_and_expansion_results(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        count = unittest.TestLoader().discover(str(ROOT / "tests"), top_level_dir=str(ROOT)).countTestCases()
        results = []
        for label, dataset in (("原题", self.report), ("留出题", self.report["held_out"])):
            configs = dataset["configs"]
            changed = set()
            for name, (_, _, k) in CONFIGS.items():
                before = configs[name]["results"]
                after = configs[f"{name}{EXPANDED_SUFFIX}"]["results"]
                for qid in before:
                    if reciprocal_rank(before[qid]["rank"], k) != reciprocal_rank(after[qid]["rank"], k):
                        changed.add(qid)
            results.append(f"{label} {len(changed)}/{len(dataset['questions'])}")

        for pattern in (r"([0-9]+) 项全部通过", r"\bAll ([0-9]+) tests pass\b"):
            with self.subTest(pattern=pattern):
                declared = re.search(pattern, readme)
                self.assertIsNotNone(declared)
                self.assertEqual(int(declared.group(1)), count)

        for expected in ("、".join(results), "原题上的收益可能偏乐观。"):
            with self.subTest(expected=expected):
                self.assertIn(expected, readme, f"README 缺少：{expected}")

    def test_eval_uses_the_configured_expansion_weight(self):
        path = self.directory / "weighted.json"
        cli("eval", "--expand", "--expansion-weight", "0.4", "--out", str(path))
        report = json.loads(path.read_bytes())
        self.assertEqual(self.report["expansion_weight"], 0.5)
        self.assertEqual(report["expansion_weight"], 0.4)
        q34 = report["configs"]["heading/bigram/1/expanded"]["results"]["Q34"]
        self.assertEqual(q34["top"][0]["chunk"], "account.md:290-374")
        self.assertAlmostEqual(q34["top"][0]["score"], 1.2)

    def test_eval_rejects_synonyms_that_are_not_lists_of_strings(self):
        data = self.directory / "invalid-synonyms"
        copytree(ROOT / "data", data)
        synonyms_path = data / "kb" / "synonyms.json"
        synonyms = json.loads(synonyms_path.read_bytes())
        output = self.directory / "invalid-synonyms.json"
        argv = ["kbeval", "eval", "--expand", "--out", str(output)]

        for invalid in ("云端日志", ["云端日志", 7]):
            with self.subTest(value=invalid):
                synonyms["遥测档案"] = invalid
                synonyms_path.write_text(json.dumps(synonyms, ensure_ascii=False), encoding="utf-8")
                with patch("kbeval.__main__.DATA", data), patch("sys.argv", argv):
                    with patch("sys.stdout", new_callable=StringIO), patch("sys.stderr", new_callable=StringIO) as stderr:
                        with self.assertRaises(SystemExit) as caught:
                            main()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("同义词条目“遥测档案”的值应为文本列表", stderr.getvalue())
                self.assertFalse(output.exists())

    def test_eval_rejects_blank_synonym_keys_before_scoring(self):
        data = self.directory / "blank-synonym-keys"
        copytree(ROOT / "data", data)
        synonyms_path = data / "kb" / "synonyms.json"
        synonyms = json.loads(synonyms_path.read_bytes())

        for index, wording in enumerate(("", " \t　")):
            with self.subTest(wording=wording):
                output = self.directory / f"blank-synonym-key-{index}.json"
                argv = ["kbeval", "eval", "--expand", "--out", str(output)]
                synonyms_path.write_text(json.dumps(synonyms | {wording: ["云端日志"]}), encoding="utf-8")
                with (
                    patch("kbeval.__main__.DATA", data),
                    patch("kbeval.__main__.build_report", wraps=build_report) as score,
                    patch("sys.argv", argv),
                    patch("sys.stdout", new_callable=StringIO) as stdout,
                    patch("sys.stderr", new_callable=StringIO) as stderr,
                ):
                    with self.assertRaises(SystemExit) as caught:
                        main()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("同义词条目名称为空", stderr.getvalue())
                score.assert_not_called()
                self.assertEqual(stdout.getvalue(), "")
                self.assertFalse(output.exists())
