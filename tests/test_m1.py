from io import StringIO
from math import log
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from kbeval.__main__ import (
    _sentence_spans, build_report, explain, export_has_path, export_has_private_text, json_strings,
    load_data, main, make_chunks, query_result,
)
from kbeval.chunk import Chunk, by_heading, fixed_width, gold_chunks
from kbeval.metrics import summarize
from kbeval.retrieve import BM25, BigramOverlap


ROOT = Path(__file__).resolve().parents[1]


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

    def test_chunk_titles_ignore_headings_inside_fenced_code(self):
        for opening, closing in (("~~~", "~~~"), ("````python\n```", "````")):
            with self.subTest(opening=opening):
                text = f"# 手册\n{opening}\n# 这是示例\n{closing}\n后续正文。\n"
                expected = Chunk("manual.md", ("手册",), 0, len(text), text)
                self.assertEqual(by_heading("manual.md", text), [expected])

                extended = text + "## 附录\n附录正文。\n"
                chunks = fixed_width("manual.md", extended, len(text), 0)
                self.assertEqual(chunks[0], expected)
                self.assertEqual(chunks[1].title_path, ("手册", "附录"))

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
        spans = _sentence_spans(documents)[gold]

        no_overlap = make_chunks(documents, "fixed-200-0")
        result = query_result(question, BM25(no_overlap).rank(question["question"]), gold_chunks(no_overlap, spans), 3)
        self.assertEqual(result["gold_chunks"], [])
        self.assertEqual(result["diagnosis"]["reason"], "答案被切断")

        with_overlap = make_chunks(documents, "fixed-200-50")
        result = query_result(question, BM25(with_overlap).rank(question["question"]), gold_chunks(with_overlap, spans), 3)
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

    def test_load_data_rejects_partial_gold_sentence(self):
        data = self.directory / "partial-gold"
        shutil.copytree(ROOT / "data", data)
        questions_file = data / "questions.jsonl"
        questions = [json.loads(line) for line in questions_file.read_text(encoding="utf-8").splitlines()]
        question = next(item for item in questions if item["id"] == "Q34")
        for gold in ("保留", "保留七天。"):
            with self.subTest(gold=gold):
                question["gold"] = gold
                questions_file.write_text("\n".join(map(json.dumps, questions)), encoding="utf-8")
                with patch("kbeval.__main__.DATA", data):
                    with self.assertRaisesRegex(ValueError, "标准原句：Q34"):
                        load_data()

    def test_load_data_checks_the_full_sentence_across_a_soft_line_break(self):
        data = self.directory / "wrapped-gold"
        shutil.copytree(ROOT / "data", data)
        doc = data / "kb" / "account.md"
        original = doc.read_text(encoding="utf-8")
        questions_file = data / "questions.jsonl"
        questions = [json.loads(line) for line in questions_file.read_text(encoding="utf-8").splitlines()]
        question = next(item for item in questions if item["id"] == "Q34")
        original_gold = question["gold"]
        cases = (
            ("\n", "云端日志仅\n保留七天。", True),
            ("\n", "保留七天。", False),
            ("\n\n", "保留七天。", True),
        )
        for gap, gold, valid in cases:
            with self.subTest(gap=gap, gold=gold):
                wrapped = f"云端日志仅{gap}保留七天。"
                doc.write_text(original.replace(original_gold, wrapped), encoding="utf-8")
                question["gold"] = gold
                questions_file.write_text("\n".join(map(json.dumps, questions)), encoding="utf-8")
                with patch("kbeval.__main__.DATA", data):
                    if valid:
                        documents, loaded = load_data()
                        self.assertIn(question, loaded)
                        self.assertTrue(gold_chunks(make_chunks(documents, "heading"), _sentence_spans(documents)[gold]))
                    else:
                        with self.assertRaisesRegex(ValueError, "标准原句：Q34"):
                            load_data()

    def test_load_data_preserves_sentence_continuations_that_look_like_list_markers(self):
        data = self.directory / "literal-list-markers"
        shutil.copytree(ROOT / "data", data)
        doc = data / "kb" / "account.md"
        original = doc.read_text(encoding="utf-8")
        questions_file = data / "questions.jsonl"
        questions = [json.loads(line) for line in questions_file.read_text(encoding="utf-8").splitlines()]
        question = next(item for item in questions if item["id"] == "Q34")
        original_gold = question["gold"]
        error_message = "设备屏幕提示\n404. 找不到该设备。"
        formula = "剩余天数 = 7\n- 已过天数。"
        for replacement, full, partial in (
            (error_message, error_message, "找不到该设备。"),
            (f"```text\n计算方法。\n{formula}\n```\n\n", formula, "已过天数。"),
        ):
            doc.write_text(original.replace(original_gold, replacement, 1), encoding="utf-8")
            for gold in (full, partial):
                with self.subTest(gold=gold):
                    question["gold"] = gold
                    questions_file.write_text("\n".join(map(json.dumps, questions)), encoding="utf-8")
                    with patch("kbeval.__main__.DATA", data):
                        if gold == full:
                            documents, loaded = load_data()
                            self.assertIn(question, loaded)
                            spans = _sentence_spans(documents)[gold]
                            self.assertTrue(gold_chunks(make_chunks(documents, "heading"), spans))
                        else:
                            with self.assertRaisesRegex(ValueError, "标准原句：Q34"):
                                load_data()

    def test_eval_finds_gold_sentence_at_its_source_range_in_a_list_item(self):
        data = self.directory / "list-item-gold"
        shutil.copytree(ROOT / "data", data)
        doc = data / "kb" / "account.md"
        original = doc.read_bytes().decode("utf-8")
        gold = "云端日志仅保留七天。"
        output = self.directory / "list-item-gold.json"
        argv = ["kbeval", "eval", "--out", str(output)]

        for marker in ("", "- ", "  1. ", "  2. ", "注意事项\n  1. ", "  1. 另一项。\n  2. "):
            with self.subTest(marker=marker):
                text = original.replace(gold, marker + gold, 1)
                doc.write_bytes(text.encode("utf-8"))
                with patch("kbeval.__main__.DATA", data), patch("sys.argv", argv):
                    with patch("sys.stdout", new_callable=StringIO):
                        main()
                report = json.loads(output.read_bytes())
                self.assertEqual(next(item["gold"] for item in report["questions"] if item["id"] == "Q34"), gold)
                result = report["configs"]["heading/bigram/1"]["results"]["Q34"]
                start, end = text.index("## 运行记录"), len(text)
                self.assertEqual(result["gold_chunks"], [f"account.md:{start}-{end}"])
                source = report["chunks"]["heading"][result["gold_chunks"][0]]
                self.assertIn(marker + gold, text[source["start"]:source["end"]])

    def test_eval_and_explain_do_not_count_opposite_sentence_as_gold(self):
        data = self.directory / "opposite-sentence"
        kb = data / "kb"
        kb.mkdir(parents=True)
        (kb / "a.md").write_text("退款规则。定制款不支持退款。", encoding="utf-8")
        (kb / "b.md").write_text("支持退款。", encoding="utf-8")
        record = {"id": "Q01", "question": "退款规则？", "gold": "支持退款。"}
        (data / "questions.jsonl").write_text(json.dumps(record), encoding="utf-8")
        output = self.directory / "opposite-sentence.json"
        config = "heading/bigram/1"

        with patch("kbeval.__main__.DATA", data):
            with patch("sys.argv", ["kbeval", "eval", "--out", str(output)]):
                with patch("sys.stdout", new_callable=StringIO):
                    main()
            with patch("sys.argv", ["kbeval", "explain", "Q01", "--config", config]):
                with patch("sys.stdout", new_callable=StringIO) as stdout:
                    main()

        row = json.loads(output.read_bytes())["configs"][config]
        result = row["results"]["Q01"]
        self.assertEqual(result["top"][0]["chunk"], "a.md:0-14")
        self.assertEqual(result["gold_chunks"], ["b.md:0-5"])
        self.assertEqual(result["rank"], 2)
        self.assertEqual(row["recall"], 0)
        self.assertIn("首个含标准原句的块排第 2 名。", stdout.getvalue())
        self.assertNotIn("命中前 1 名。", stdout.getvalue())

    def test_eval_matrix_and_gold_are_independent_of_retriever(self):
        documents, questions = load_data()
        gold = "云端日志仅保留七天。"
        documents["returns.md"] += f"\n不能据此断言{gold}\n"
        report = build_report(documents, questions, include_text=False)
        sentence_spans = _sentence_spans(documents)
        gold_start = documents["account.md"].index(gold)
        self.assertEqual(sentence_spans[gold], {"account.md": [(gold_start, gold_start + len(gold))]})

        self.assertEqual(len(documents), 6)
        self.assertEqual(len(questions), 40)
        self.assertEqual(report["questions"], questions)
        chunkings = ("heading", "fixed-200-0", "fixed-200-50")
        configs = report["configs"]
        self.assertEqual(set(configs), {
            f"{chunking}/{retriever}/{k}" for chunking in chunkings
            for retriever in ("bm25", "bigram") for k in (1, 3, 5)
        })
        for chunking in chunkings:
            chunks = report["chunks"][chunking]
            for retriever in ("bm25", "bigram"):
                for k in (1, 3, 5):
                    row = configs[f"{chunking}/{retriever}/{k}"]
                    self.assertEqual(row["chunk_count"], len(chunks))
                    self.assertEqual(set(row["results"]), {item["id"] for item in questions})
                    for item in questions:
                        spans = sentence_spans[item["gold"]]
                        expected = sorted(
                            cid for cid, chunk in chunks.items()
                            if any(
                                chunk["start"] <= start and end <= chunk["end"]
                                for start, end in spans.get(chunk["document"], ())
                            )
                        )
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

        documents = {"a.md": "七天内可办理。", "b.md": "撤单请登记。", "c.md": "撤件请寄回。"}
        questions = [
            {"id": "Q01", "question": "申请还能撤回吗？", "gold": documents["a.md"]},
            {"id": "Q02", "question": "几天内可办理？", "gold": documents["a.md"]},
            {"id": "Q03", "question": "寄到哪？", "gold": documents["c.md"]},
        ]
        for order, records in (("forward", questions), ("reverse", questions[::-1])):
            with self.subTest(order=order):
                report = build_report(documents, records, include_text=False)
                bm25 = report["configs"]["heading/bm25/3"]
                bigram = report["configs"]["heading/bigram/3"]
                self.assertEqual(bm25["mrr"], bigram["mrr"])
                self.assertEqual(bm25["mrr"], 7 / 9)
                self.assertEqual(report["recommendation"]["config"], "heading/bm25/3")

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

    def test_compare_validates_result_and_rank_before_printing(self):
        config, qid = "heading/bigram/1", "Q01"
        report = json.loads(self.baseline.read_bytes())
        results = report["configs"][config]["results"]
        self.assertEqual(results[qid]["rank"], 1)
        right = self.directory / "changed-result.json"

        for row, message in (
            ({"rank": 0.5}, "名次无效，须为正整数或 null"),
            (None, "结果须为 JSON 对象"),
            ([], "结果须为 JSON 对象"),
        ):
            with self.subTest(row=row):
                results[qid] = row
                right.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaises(subprocess.CalledProcessError) as caught:
                    cli("compare", str(self.baseline), str(right))
                error = caught.exception
                self.assertEqual(error.returncode, 2, error.stderr)
                self.assertEqual(error.stdout, "")
                self.assertIn(f"{config} 的 {qid} {message}", error.stderr)
                self.assertNotIn("Traceback", error.stderr)

        results[qid] = {"rank": None}
        right.write_text(json.dumps(report), encoding="utf-8")
        self.assertIn("变好 0，变坏 1，不变 719。", cli("compare", str(self.baseline), str(right)))

    def test_compare_rejects_question_missing_from_both_configs(self):
        config = "heading/bigram/3"
        report = json.loads(self.baseline.read_bytes())
        del report["configs"][config]["results"]["Q34"]
        left = self.directory / "missing-before.json"
        right = self.directory / "missing-after.json"
        left.write_text(json.dumps(report), encoding="utf-8")
        report["configs"]["heading/bm25/1"]["results"]["Q01"]["rank"] = None
        right.write_text(json.dumps(report), encoding="utf-8")

        with self.assertRaises(subprocess.CalledProcessError) as caught:
            cli("compare", str(left), str(right))
        self.assertEqual(caught.exception.returncode, 2)
        self.assertEqual(caught.exception.stdout, "")
        self.assertIn(config, caught.exception.stderr)
        self.assertIn("Q34", caught.exception.stderr)

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

    def test_explain_reports_zero_score_tie_at_cutoff(self):
        documents = {"b.md": "水杯。", "a.md": "奶茶。"}
        question = {"id": "Q01", "question": "水箱", "gold": "水杯。"}
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            explain(documents, question, "heading/bigram/1")

        output = stdout.getvalue()
        self.assertIn("1. a.md", output)
        self.assertIn("首个含标准原句的块排第 2 名。", output)
        self.assertIn("归因：同分，按文件名和块起点排序", output)
        self.assertIn("第 1 名与首个含原句块得分相同：0.0000", output)

    def test_explain_bigram_miss_lists_higher_scoring_chunk(self):
        documents = {"a.md": "Refunds 是栏目名。", "b.md": "可申请 refund。"}
        question = {"id": "Q01", "question": "refunds?", "gold": documents["b.md"]}
        chunks = make_chunks(documents, "heading")
        ranking = BigramOverlap(chunks).rank(question["question"])
        self.assertEqual([(chunk.id, score) for chunk, score in ranking], [
            ("a.md:0-13", 6), ("b.md:0-11", 5),
        ])

        with patch("sys.stdout", new_callable=StringIO) as stdout:
            explain(documents, question, "heading/bigram/1")

        output = stdout.getvalue()
        self.assertIn("首个含标准原句的块排第 2 名。", output)
        self.assertIn("归因：被相似段落挤掉", output)
        self.assertIn("排在首个含原句块之前的块：a.md:0-13", output)

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

    def test_export_allows_slash_in_prose_and_public_urls(self):
        data = self.directory / "public-demo-corpus"
        shutil.copytree(ROOT / "data", data)
        doc = data / "kb" / "smart.md"
        original = doc.read_text(encoding="utf-8") + "\n"
        url_host = "https://support.example.com"
        padding = " " * (-(len(original) + len(url_host)) % 200)
        cases = ("请查看配送/安装说明。", f"请看[帮助]({url_host}/help)。", f"{padding}{url_host}/help")
        for index, text in enumerate(cases):
            with self.subTest(text=text.strip()):
                doc.write_text(original + text + "\n", encoding="utf-8")
                output = self.directory / f"public-demo-{index}.json"
                with (
                    patch("kbeval.__main__.DATA", data),
                    patch("sys.argv", ["kbeval", "export", "--out", str(output)]),
                    patch("sys.stdout", new_callable=StringIO),
                ):
                    main()
                chunks = json.loads(output.read_bytes())["chunks"]["heading"].values()
                self.assertTrue(any(text.strip() in chunk["text"] for chunk in chunks))

    def test_export_rejects_absolute_path_before_writing(self):
        data = self.directory / "path-in-corpus"
        shutil.copytree(ROOT / "data", data)
        doc = data / "kb" / "smart.md"
        original = doc.read_text(encoding="utf-8")
        cases = (
            "见 `/tmp/demo.md`。", "见/tmp/demo.md。", "示例 `cc -I/tmp/demo/include`。",
            "见~/demo/手册.md。", r"见C:\Users\demo\manual.md。",
        )
        for index, text in enumerate(cases):
            with self.subTest(text=text):
                doc.write_text(f"{original}\n{text}\n", encoding="utf-8")
                output = self.directory / f"path-in-demo-{index}.json"
                with (
                    patch("kbeval.__main__.DATA", data),
                    patch("sys.argv", ["kbeval", "export", "--out", str(output)]),
                    patch("sys.stdout", new_callable=StringIO) as stdout,
                    patch("sys.stderr", new_callable=StringIO) as stderr,
                ):
                    with self.assertRaises(SystemExit) as caught:
                        main()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("绝对路径", stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                self.assertFalse(output.exists())

    def test_export_rejects_private_text_before_writing(self):
        data = self.directory / "private-text-in-corpus"
        shutil.copytree(ROOT / "data", data)
        doc = data / "kb" / "account.md"
        original = doc.read_text(encoding="utf-8")
        for label, private_text in (
            ("email", "demo@example.invalid"),
            ("plain-phone", "13900000000"),
            ("hyphenated-phone", "139-0000-0000"),
        ):
            with self.subTest(value=label):
                doc.write_text(f"{original}\n请联系{private_text}确认。\n", encoding="utf-8")
                output = self.directory / f"private-text-in-demo-{label}.json"
                with (
                    patch("kbeval.__main__.DATA", data),
                    patch("sys.argv", ["kbeval", "export", "--out", str(output)]),
                    patch("sys.stdout", new_callable=StringIO) as stdout,
                    patch("sys.stderr", new_callable=StringIO) as stderr,
                ):
                    with self.assertRaises(SystemExit) as caught:
                        main()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("演示数据含有隐私信息", stderr.getvalue())
                self.assertNotIn(private_text, stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                self.assertFalse(output.exists())

    def test_export_privacy_check_rejects_paths_and_phones_in_prose(self):
        for text in ("详见 `/tmp/demo.md`。", "请联系13900000000确认。"):
            with self.subTest(text=text):
                payload = json.loads(json.dumps({"nested": [{"detail": text}]}))
                with self.assertRaisesRegex(AssertionError, "发现路径或隐私信息"):
                    self.assert_export_has_no_private_text(payload)

    def assert_export_has_no_private_text(self, export):
        self.assertFalse(export_has_path(export), "发现路径或隐私信息")
        self.assertFalse(export_has_private_text(export), "发现路径或隐私信息")
        for value in json_strings(export):
            self.assertNotIn(str(ROOT), value)
            self.assertNotIn(str(self.directory), value)
