import argparse
import json
from pathlib import Path
import re

from .chunk import Chunk, by_heading, fixed_width, gold_chunks
from .diagnose import explain_miss
from .metrics import reciprocal_rank, summarize
from .retrieve import BM25, DEFAULT_EXPANSION_WEIGHT, BigramOverlap, Ranked, expand_query


DATA = Path(__file__).resolve().parents[1] / "data"
CHUNKINGS = {"heading": None, "fixed-200-0": 0, "fixed-200-50": 50}
RETRIEVERS = {"bm25": BM25, "bigram": BigramOverlap}
CUTOFFS = (1, 3, 5)
CONFIGS = {
    f"{chunking}/{retriever}/{k}": (chunking, retriever, k)
    for chunking in CHUNKINGS
    for retriever in RETRIEVERS
    for k in CUTOFFS
}
EXPANDED_SUFFIX = "/expanded"
ALL_CONFIGS = CONFIGS | {f"{name}{EXPANDED_SUFFIX}": config for name, config in CONFIGS.items()}
_EXPORT_LOCATION = re.compile(
    r"https?://[^\s<>`\"'()、，。！？；]+"
    r"|(?P<local_path>"
    r"(?:(?<![\w./\\~-])(?:-[A-Za-z])?/+"
    r"|(?<![A-Za-z0-9_./\\~-])(?:-[A-Za-z])?(?:~[/\\]|[A-Za-z]:[/\\]|\\\\|/+(?=[\w.-]+/)))"
    r"[^\s/\\])",
    re.IGNORECASE,
)
_SENTENCE_BLOCK_BREAK = re.compile(
    r"(\r?\n[ \t]*\r?\n|^ {0,3}#{1,6}(?=[ \t]|\r?$)[^\r\n]*)", re.MULTILINE,
)


def load_data() -> tuple[dict[str, str], list[dict[str, str]]]:
    documents = {
        path.name: path.read_bytes().decode("utf-8")
        for path in sorted((DATA / "kb").glob("*.md"))
    }
    return documents, load_questions(documents, DATA / "questions.jsonl")


def load_questions(documents: dict[str, str], path: Path) -> list[dict[str, str]]:
    sentences = {
        match.group().strip()
        for text in documents.values()
        for block in _SENTENCE_BLOCK_BREAK.split(text)
        for match in re.finditer(r"[^。！？!?]+[。！？!?]+", block)
    }
    questions = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        qid, question, gold = (record[key] for key in ("id", "question", "gold"))
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError(f"题号无效或重复：{qid}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"题目为空：{qid}")
        if not isinstance(gold, str) or gold not in sentences:
            raise ValueError(f"语料中找不到标准原句：{qid}")
        seen.add(qid)
        questions.append({"id": qid, "question": question, "gold": gold})
    if not questions:
        raise ValueError("题库为空")
    return questions


def make_chunks(documents: dict[str, str], chunking: str) -> list[Chunk]:
    overlap = CHUNKINGS[chunking]
    chunks = []
    for document, text in documents.items():
        if overlap is None:
            pieces = by_heading(document, text)
        else:
            pieces = fixed_width(document, text, 200, overlap)
        chunks.extend(pieces)
    return chunks


def query_result(record: dict[str, str], ranked: Ranked, gold_ids: set[str], k: int, *, expansion: str = "") -> dict:
    rank = next((index for index, (chunk, _) in enumerate(ranked, 1) if chunk.id in gold_ids), None)
    diagnosis = None
    if rank is None or rank > k:
        question = f"{record['question']} {expansion}" if expansion else record["question"]
        diagnosis = explain_miss(question, record["gold"], ranked, rank, k)
    return {
        "rank": rank,
        "gold_chunks": sorted(gold_ids),
        "top": [{"chunk": chunk.id, "score": score} for chunk, score in ranked[:k]],
        "diagnosis": diagnosis,
    }


def chunk_metadata(chunk: Chunk) -> dict:
    return {"document": chunk.document, "title_path": chunk.title_path, "start": chunk.start, "end": chunk.end}


def build_report(
    documents: dict[str, str], questions: list[dict[str, str]], *,
    include_text: bool, synonyms: dict[str, list[str]] | None = None,
    expansion_weight: float = DEFAULT_EXPANSION_WEIGHT,
    held_out: list[dict[str, str]] | None = None,
) -> dict:
    configs = {}
    held_out_configs = {}
    datasets = [(questions, configs)]
    if held_out is not None:
        datasets.append((held_out, held_out_configs))

    chunk_index = {}
    for chunking in CHUNKINGS:
        chunks = make_chunks(documents, chunking)
        chunk_index[chunking] = {chunk.id: chunk_metadata(chunk) for chunk in chunks}
        if include_text:
            for chunk in chunks:
                chunk_index[chunking][chunk.id]["text"] = chunk.text

        indexes = {name: index_type(chunks) for name, index_type in RETRIEVERS.items()}
        for records, dataset_configs in datasets:
            matches = {item["id"]: gold_chunks(chunks, item["gold"]) for item in records}
            modes = {"": dict.fromkeys(matches, "")}
            if synonyms is not None:
                modes[EXPANDED_SUFFIX] = {item["id"]: expand_query(item["question"], synonyms) for item in records}
            for suffix, expansions in modes.items():
                for retriever, index in indexes.items():
                    rankings = {
                        item["id"]: index.rank(
                            item["question"], expansion=expansions[item["id"]], expansion_weight=expansion_weight,
                        )
                        for item in records
                    }
                    for k in CUTOFFS:
                        results = {
                            item["id"]: query_result(
                                item, rankings[item["id"]], matches[item["id"]], k,
                                expansion=expansions[item["id"]] if expansion_weight else "",
                            )
                            for item in records
                        }
                        ranks = {qid: result["rank"] for qid, result in results.items()}
                        dataset_configs[f"{chunking}/{retriever}/{k}{suffix}"] = {
                            "chunk_count": len(chunks), **summarize(ranks, k), "results": results,
                        }

    candidates = ((name, row) for name, row in configs.items() if ALL_CONFIGS[name][2] == 3)
    winner, row = max(
        candidates, key=lambda entry: (entry[1]["recall"], entry[1]["mrr"], -entry[1]["chunk_count"]),
    )
    reason = (
        "按 Recall@3 降序、MRR 降序、块数升序选择，三项相同则取矩阵中先出现的设置。"
        f"该设置的 Recall@3={row['recall']:.4f}，MRR={row['mrr']:.5f}，共 {row['chunk_count']} 块。"
        "结论仅适用于这组虚构数据。"
    )
    report = {
        "questions": questions,
        "chunks": chunk_index,
        "configs": configs,
    }
    if synonyms is not None:
        report["expansion_weight"] = expansion_weight
    if held_out is not None:
        report["held_out"] = {"questions": held_out, "configs": held_out_configs}
    report["recommendation"] = {"config": winner, "reason": reason}
    return report


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def export_has_path(report: dict) -> bool:
    return any(
        match.group("local_path") is not None
        for value in json_strings(report)
        for match in _EXPORT_LOCATION.finditer(value)
    )


def validate_result_ids(name: str, expected_ids: set[str], *results: dict) -> None:
    different_ids = set()
    for result in results:
        different_ids.update(result.keys() ^ expected_ids)
    if different_ids:
        raise ValueError(f"{name} 的结果与题目清单不符：{', '.join(sorted(different_ids))}")


def validate_comparison(left: dict, right: dict) -> None:
    if left["configs"].keys() != right["configs"].keys():
        raise ValueError("两份报告的评测配置不同")
    left_questions, right_questions = (
        {item["id"]: (item["question"], item["gold"]) for item in report["questions"]}
        for report in (left, right)
    )
    different_ids = left_questions.keys() ^ right_questions.keys()
    if different_ids:
        raise ValueError(f"两份报告的题号不同：{', '.join(sorted(different_ids))}")
    changed = [qid for qid in sorted(left_questions) if left_questions[qid] != right_questions[qid]]
    if changed:
        raise ValueError(f"两份报告的问题或标准原句不同：{', '.join(changed)}")

    expected_ids = set(left_questions)
    for name, before in left["configs"].items():
        after = right["configs"][name]
        validate_result_ids(name, expected_ids, before["results"], after["results"])


def compare_reports(left: dict, right: dict) -> None:
    has_held_out = "held_out" in left
    if has_held_out != ("held_out" in right):
        missing = "第二份" if has_held_out else "第一份"
        raise ValueError(f"{missing}报告缺少留出题（held_out）")
    validate_comparison(left, right)
    if has_held_out:
        left_held_out, right_held_out = left["held_out"], right["held_out"]
        validate_comparison(left_held_out, right_held_out)
        print("原有题")
        compare_configs(left["configs"], right["configs"])
        print("留出题")
        compare_configs(left_held_out["configs"], right_held_out["configs"])
    else:
        compare_configs(left["configs"], right["configs"])


def compare_configs(before_configs: dict, after_configs: dict, *, show_displacement: bool = False) -> None:
    better_count = worse_count = same_count = 0
    for name, before in before_configs.items():
        k = ALL_CONFIGS[name][2]
        before, after = before["results"], after_configs[name]["results"]
        better, worse = [], []
        for qid in sorted(before):
            old = reciprocal_rank(before[qid]["rank"], k)
            new = reciprocal_rank(after[qid]["rank"], k)
            if new > old:
                better.append(qid)
            elif new < old:
                worse.append(qid)
        same = len(before) - len(better) - len(worse)
        better_count += len(better)
        worse_count += len(worse)
        same_count += same
        if better or worse:
            print(f"{name}｜变好 {len(better)} 题：{', '.join(better) or '无'}"
                  f"｜变坏 {len(worse)} 题：{', '.join(worse) or '无'}｜不变 {same} 题")
        if show_displacement:
            for qid in worse:
                old_rank, new_rank = before[qid]["rank"], after[qid]["rank"]
                old_ahead = {hit["chunk"] for hit in before[qid]["top"][:old_rank - 1]}
                new_ahead = after[qid]["top"][:new_rank - 1] if new_rank is not None else after[qid]["top"]
                passed = [hit["chunk"] for hit in new_ahead if hit["chunk"] not in old_ahead]
                print(f"  {qid}｜第 {old_rank} 名 → 第 {new_rank} 名｜新超过标准答案块：{', '.join(passed)}")
    total = better_count + worse_count + same_count
    print(f"{len(before_configs)} 格，共 {total} 项逐题对比："
          f"变好 {better_count}，变坏 {worse_count}，不变 {same_count}。")


def compare_expansion(report: dict) -> None:
    if "held_out" not in report:
        raise ValueError("单份报告比较需要用 eval --expand 生成报告")
    datasets = (("原有题", report), ("留出题", report["held_out"]))
    for _, dataset in datasets:
        expected_ids = {item["id"] for item in dataset["questions"]}
        for name in ALL_CONFIGS:
            validate_result_ids(name, expected_ids, dataset["configs"][name]["results"])

    print(f"同义词扩展关闭 → 开启（权重 {report['expansion_weight']:g}）")
    for label, dataset in datasets:
        configs = dataset["configs"]
        print(f"{label}（{len(dataset['questions'])} 题）")
        compare_configs(
            {name: configs[name] for name in CONFIGS},
            {name: configs[f"{name}{EXPANDED_SUFFIX}"] for name in CONFIGS},
            show_displacement=True,
        )


def explain(documents: dict[str, str], record: dict[str, str], config: str) -> None:
    chunking, retriever, k = CONFIGS[config]
    chunks = make_chunks(documents, chunking)
    lookup = {chunk.id: chunk for chunk in chunks}
    ranked = RETRIEVERS[retriever](chunks).rank(record["question"])
    result = query_result(record, ranked, gold_chunks(chunks, record["gold"]), k)
    print(f"{record['id']}｜{config}｜{record['question']}")
    print(f"标准原句：{record['gold']}")
    for pos, hit in enumerate(result["top"], 1):
        chunk = lookup[hit["chunk"]]
        title = " / ".join(chunk.title_path) or "无标题"
        marker = "｜含标准原句" if chunk.id in result["gold_chunks"] else ""
        print(f"{pos}. {chunk.document} [{chunk.start}, {chunk.end})｜{title}｜得分 {hit['score']:.4f}{marker}")
        excerpt = " ".join(chunk.text.split())
        print(f"   {excerpt[:90]}{'…' if len(excerpt) > 90 else ''}")
    rank = result["rank"]
    if rank is not None:
        print(f"首个含标准原句的块排第 {rank} 名。")
    diagnosis = result["diagnosis"]
    if diagnosis is None:
        print(f"命中前 {k} 名。")
        return
    print(f"归因：{diagnosis['reason']}")
    if "tied_score" in diagnosis:
        print(f"第 {k} 名与首个含原句块得分相同：{diagnosis['tied_score']:.4f}")
    if "question_terms" in diagnosis:
        print(f"问题词：{', '.join(diagnosis['question_terms']) or '无'}")
        print(f"原句词：{', '.join(diagnosis['gold_terms']) or '无'}")
    if "ahead" in diagnosis:
        print(f"排在首个含原句块之前的块：{', '.join(diagnosis['ahead'])}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python3 -m kbeval", description="知识库检索评测")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--out", type=Path, required=True)
    evaluation.add_argument("--expand", action="store_true", help="同时评测关闭和开启同义词扩展")
    evaluation.add_argument("--expansion-weight", type=float, help="扩展词权重，默认 0.5")
    commands.add_parser("export").add_argument("--out", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("a", type=Path)
    comparison.add_argument("b", type=Path, nargs="?")
    explanation = commands.add_parser("explain")
    explanation.add_argument("qid")
    explanation.add_argument("--config", choices=tuple(CONFIGS), required=True)
    args = parser.parse_args()

    try:
        if args.command == "compare":
            first = read_report(args.a)
            if args.b is None:
                compare_expansion(first)
            else:
                compare_reports(first, read_report(args.b))
            return
        documents, questions = load_data()
        if args.command == "explain":
            record = next((record for record in questions if record["id"] == args.qid), None)
            if record is None:
                raise ValueError(f"找不到题号：{args.qid}")
            explain(documents, record, args.config)
            return

        export = args.command == "export"
        # Keep the context of URLs split by fixed-width chunks.
        if export and export_has_path({"documents": documents, "questions": questions}):
            raise ValueError("演示数据含有本机绝对路径")
        if args.command == "eval" and args.expand:
            weight = DEFAULT_EXPANSION_WEIGHT if args.expansion_weight is None else args.expansion_weight
            if not 0 <= weight < 1:
                raise ValueError("扩展权重须在 0 到 1 之间，不含 1")
            synonyms = read_report(DATA / "kb" / "synonyms.json")
            held_out = load_questions(documents, DATA / "held_out.jsonl")
            report = build_report(documents, questions, include_text=False, synonyms=synonyms,
                                  expansion_weight=weight, held_out=held_out)
        else:
            if args.command == "eval" and args.expansion_weight is not None:
                raise ValueError("设置扩展权重时须同时使用 --expand")
            report = build_report(documents, questions, include_text=export)
        args.out.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        if export:
            print(f"已写出演示数据：{args.out}")
        else:
            question_count = f"{len(questions)} 题"
            if "held_out" in report:
                question_count = f"原有 {question_count}及留出 {len(report['held_out']['questions'])} 题"
            print(f"已评测 {len(report['configs'])} 格、{question_count}，报告：{args.out}")
            recommendation = report["recommendation"]
            print(f"推荐 {recommendation['config']}。{recommendation['reason']}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"错误：{error}\n")


if __name__ == "__main__":
    main()
