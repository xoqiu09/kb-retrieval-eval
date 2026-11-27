import argparse
import json
from pathlib import Path

from .chunk import Chunk, by_heading, fixed_width, gold_chunks
from .diagnose import explain_miss
from .metrics import reciprocal_rank, summarize
from .retrieve import BM25, BigramOverlap, Ranked


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


def load_data() -> tuple[dict[str, str], list[dict[str, str]]]:
    documents = {
        path.name: path.read_bytes().decode("utf-8")
        for path in sorted((DATA / "kb").glob("*.md"))
    }
    questions = []
    seen = set()
    for line in (DATA / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        qid, question, gold = (record[key] for key in ("id", "question", "gold"))
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError(f"题号无效或重复：{qid}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"题目为空：{qid}")
        if not isinstance(gold, str) or not gold.strip() or not any(
            gold in text for text in documents.values()
        ):
            raise ValueError(f"语料中找不到标准原句：{qid}")
        seen.add(qid)
        questions.append({"id": qid, "question": question, "gold": gold})
    if not questions:
        raise ValueError("题库为空")
    return documents, questions


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


def query_result(record: dict[str, str], ranked: Ranked, gold_ids: set[str], k: int) -> dict:
    rank = next((index for index, (chunk, _) in enumerate(ranked, 1) if chunk.id in gold_ids), None)
    diagnosis = None
    if rank is None or rank > k:
        diagnosis = explain_miss(record["question"], record["gold"], ranked, rank)
    return {
        "rank": rank,
        "gold_chunks": sorted(gold_ids),
        "top": [{"chunk": chunk.id, "score": score} for chunk, score in ranked[:k]],
        "diagnosis": diagnosis,
    }


def chunk_metadata(chunk: Chunk) -> dict:
    return {"document": chunk.document, "title_path": chunk.title_path, "start": chunk.start, "end": chunk.end}


def build_report(documents: dict[str, str], questions: list[dict[str, str]], *, include_text: bool) -> dict:
    configs = {}
    chunk_index = {}
    for chunking in CHUNKINGS:
        chunks = make_chunks(documents, chunking)
        chunk_index[chunking] = {chunk.id: chunk_metadata(chunk) for chunk in chunks}
        if include_text:
            for chunk in chunks:
                chunk_index[chunking][chunk.id]["text"] = chunk.text

        matches = {item["id"]: gold_chunks(chunks, item["gold"]) for item in questions}
        for retriever, index_type in RETRIEVERS.items():
            index = index_type(chunks)
            rankings = {item["id"]: index.rank(item["question"]) for item in questions}
            for k in CUTOFFS:
                results = {}
                for item in questions:
                    qid = item["id"]
                    results[qid] = query_result(item, rankings[qid], matches[qid], k)
                ranks = {qid: result["rank"] for qid, result in results.items()}
                configs[f"{chunking}/{retriever}/{k}"] = {
                    "chunk_count": len(chunks), **summarize(ranks, k), "results": results,
                }

    candidates = ((name, configs[name]) for name, (_, _, k) in CONFIGS.items() if k == 3)
    winner, row = max(
        candidates, key=lambda entry: (entry[1]["recall"], entry[1]["mrr"], -entry[1]["chunk_count"]),
    )
    reason = (
        "按 Recall@3 降序、MRR 降序、块数升序选择，三项相同则取矩阵中先出现的设置。"
        f"该设置的 Recall@3={row['recall']:.4f}，MRR={row['mrr']:.5f}，共 {row['chunk_count']} 块。"
        "结论仅适用于这组虚构数据。"
    )
    return {
        "questions": questions,
        "chunks": chunk_index,
        "configs": configs,
        "recommendation": {"config": winner, "reason": reason},
    }


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_reports(left: dict, right: dict) -> None:
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

    better_count = worse_count = same_count = 0
    for name, before in left["configs"].items():
        k = CONFIGS[name][2]
        before, after = before["results"], right["configs"][name]["results"]
        if before.keys() != after.keys():
            raise ValueError(f"两份报告在 {name} 的题号不同")
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
    total = better_count + worse_count + same_count
    print(f"{len(left['configs'])} 格，共 {total} 项逐题对比："
          f"变好 {better_count}，变坏 {worse_count}，不变 {same_count}。")


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
    if "question_terms" in diagnosis:
        print(f"问题词：{', '.join(diagnosis['question_terms']) or '无'}")
        print(f"原句词：{', '.join(diagnosis['gold_terms']) or '无'}")
    if "ahead" in diagnosis:
        print(f"排在首个含原句块之前的块：{', '.join(diagnosis['ahead'])}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python3 -m kbeval", description="知识库检索评测")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("eval", "export"):
        commands.add_parser(command).add_argument("--out", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("a", type=Path)
    comparison.add_argument("b", type=Path)
    explanation = commands.add_parser("explain")
    explanation.add_argument("qid")
    explanation.add_argument("--config", choices=tuple(CONFIGS), required=True)
    args = parser.parse_args()

    try:
        if args.command == "compare":
            compare_reports(read_report(args.a), read_report(args.b))
            return
        documents, questions = load_data()
        if args.command == "explain":
            record = next((record for record in questions if record["id"] == args.qid), None)
            if record is None:
                raise ValueError(f"找不到题号：{args.qid}")
            explain(documents, record, args.config)
            return

        export = args.command == "export"
        report = build_report(documents, questions, include_text=export)
        args.out.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        if export:
            print(f"已写出演示数据：{args.out}")
        else:
            print(f"已评测 {len(report['configs'])} 格、{len(questions)} 题，报告：{args.out}")
            recommendation = report["recommendation"]
            print(f"推荐 {recommendation['config']}。{recommendation['reason']}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"错误：{error}\n")


if __name__ == "__main__":
    main()
