# 知识库检索评测

客服团队调整知识库搜索后，需要知道坐席能否找到正确的退换货或保修条款，以及哪些问题变得更难查。

- 用带标准原句的客服问题，对比文档分段和搜索方法。
- 看哪种设置更常找到正确段落，逐题列出哪些变好、哪些变差。
- 解释为什么漏掉答案，并标出搜索结果在原文中的位置。

## 怎么验证的

运行 `python3 -m unittest`，47 项全部通过。测试用手算核对命中率和排名指标，也验证分段重叠能找回被切断的答案。

在隔离副本里故意取消分段重叠、改错排名公式、让 BM25 忽略段落长度，对应测试都按预期失败。

开启同义词扩展后，按题号去重，至少在一组设置中前 k 名的命中情况或名次有变化的是原题 3/40、留出题 0/8。写词表时参考过原题的漏检案例，原题上的收益可能偏乐观。

## 快速运行

[在线演示](https://xoqiu.cn/retrieval-demo/index.html) · [GitHub 仓库](https://github.com/xoqiu09/kb-retrieval-eval)

克隆后在仓库根目录运行。第一条生成评测报告，第二条解释一道漏检。

```sh
python3 -m kbeval eval --out ./retrieval-report.json
python3 -m kbeval explain Q07 --config fixed-200-0/bm25/3
```

数据全是虚构的，未接入真实模型，也没有真实用户参与。我负责设定需求和验收标准，指导 AI 编程工具实现。

## English overview

Support teams need to know whether search returns the policy text their agents need.\
The tool compares document splitting and search methods on fictional support questions.\
It shows which questions improve or regress, explains misses, and links results to the source text.\
All 47 tests pass on the current version. Deliberate code changes triggered the expected test failures.\
I set the requirements and acceptance criteria and directed AI coding tools to build it. No live model or real users were involved.

## 怎么运行

从仓库根目录运行。使用 Python 3.10 或更高版本，自带标准库即可离线运行。下面的 JSON 写到 macOS 和 Linux 的 `/tmp`，可以把 `--out` 换成你有写权限的位置。

跑两次完整评测，分别留下报告。每份报告包含 3 种切块方式、2 种检索器、3 个排名截点，共 18 格。`configs` 下有每格的 Recall、MRR、靠后命中和逐题结果，文件末尾是推荐设置及理由。

```sh
python3 -m kbeval eval --out /tmp/kbeval-w19p1-a.json
python3 -m kbeval eval --out /tmp/kbeval-w19p1-b.json
```

逐题比较两份报告。第二份在前 k 名内排得更高或从未命中变成命中，算变好。反过来算变坏。两次都在前 k 名之外算不变。示例用同一份语料连续跑两次，结果应该全部不变。有改动时可替换为改动前后的报告。两份报告需使用相同的配置和题号，每个题号的问题和标准原句也必须相同。

```sh
python3 -m kbeval compare /tmp/kbeval-w19p1-a.json /tmp/kbeval-w19p1-b.json
```

[快速运行](#快速运行)中的 Q07 示例显示原句被切断。改用下面的配置保留 50 字重叠，完整原句会回到结果里。配置写作 `切块/检索器/k`，取值分别为 `heading`、`fixed-200-0`、`fixed-200-50`，`bm25`、`bigram`，以及 `1`、`3`、`5`。

```sh
python3 -m kbeval explain Q07 --config fixed-200-50/bm25/3
```

解释扩展配置时，将报告中的 `expansion_weight` 传给 `--expansion-weight`，以复现原报告的排名。下面以权重 `0.2` 为例。

```sh
python3 -m kbeval explain Q34 --config fixed-200-50/bm25/3/expanded --expansion-weight 0.2
```

导出网站演示页可用的数据。导出包含评测结果和虚构块原文，来源字段只写文档文件名，重复导出逐字节一致。

```sh
python3 -m kbeval export --out /tmp/kbeval-w19p1-demo.json
```

运行验收测试。

```sh
python3 -m unittest -v
```

## 结果怎么看

- 标题切块以 Markdown 的 `#` 系列标题为边界。定长切块按 200 个 Unicode 字符滑动，标题路径取块起点处的路径。位置从 0 开始，含起点、不含终点。
- BM25 用中文单字和相邻两字作为词，英文和数字串转小写后按整词计分。二元组检索数问题和块共有的不同相邻字符对。两者都对所有块打分，包括零分块。同分按文档名和块起点排序。
- 每道题取第一个完整包含原句的块。Recall@k 是它出现在前 k 名的题所占比例。MRR 是所有题的倒数排名平均值，排名取自全部结果，无完整块的题计 0。靠后命中指排第 2 至第 k 名的题。
- 未命中时先检查有没有块完整包含原句。没有就归为“答案被切断”。如果首个含原句的块排在前 k 名之外，且得分低于第 k 名，就按当前检索器的规则检查问题与原句是否重合，BM25 用分词，bigram 用相邻字符对。没有重合就列出两组词或字符对，有重合就列出排在首个含原句块之前的块。
- 先比 Recall@3，再比 MRR，最后选块数较少的设置。三项都相同时按矩阵顺序取第一项。`heading/bigram/3` 和 `fixed-200-50/bm25/3` 都命中 39/40 题，前者的 MRR 为 0.97708，后者为 0.97667，因此推荐前者。推荐只适用于这组虚构数据。

## 每项测试证明什么

- `test_heading_paths_and_source_ranges` 逐块检查嵌套标题的路径和对应原文。
- `test_fixed_windows_keep_exact_overlap_and_tail` 检查 0 和 50 字重叠、块起点的标题路径，以及最后一块保留末字。
- `test_q07_overlap_recovers_a_split_gold_sentence` 检查同一原句在无重叠时被切断，重叠 50 字后能被完整检出。
- `test_bm25_matches_three_document_hand_calculation` 用三个短文档的手算分数和顺序检查 BM25，短文档应排在词出现次数更多的长文档之前。
- `test_recall_and_mrr_match_hand_calculation` 检查 k 为 1、3、5 时的手算 Recall、MRR 和靠后命中，确认 MRR 使用完整排名。
- `test_eval_matrix_and_gold_are_independent_of_retriever` 检查 6 篇文档、40 题、18 格，按原文范围重算标准块，并核对两种检索器对 Q03 的不同排名。
- `test_eval_uses_the_requested_recall_cutoff` 检查同一排名在不同 k 下的报告 Recall。
- `test_recommendation_prefers_mrr_when_recall_ties` 检查 Recall@3 相同时，MRR 较高的设置优先，即使它的块数更多。
- `test_eval_is_byte_identical_after_rerun` 用两次 CLI 评测检查报告逐字节一致。
- `test_compare_reports_changes_and_counts` 检查 k 为 1 和 3 时变好、变坏和不变的题数及变化题号。
- `test_compare_rejects_changed_question_or_gold` 检查同一题号的问题或标准原句变化时，终端报出题号并停止比较。
- `test_explain_prints_each_diagnosis_with_evidence` 检查三类归因及终端给出的词或排在前面的块。
- `test_export_is_stable_and_uses_only_public_corpus_sources` 检查重复导出、块原文来源，以及 JSON 文本里的绝对路径和常见隐私信息。
- `test_export_privacy_check_rejects_paths_and_phones_in_prose` 确认隐私检查能识别句中绝对路径和紧贴汉字的合成手机号。

## 本里程碑的范围

只评测检索。未接入真实模型和向量库，未做网页或答案生成。网站演示页可以另行读取导出的 JSON。
