from bisect import bisect_right
from dataclasses import dataclass
import re


_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+([^\r\n]*)|(?=\r?$))", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    document: str
    title_path: tuple[str, ...]
    start: int
    end: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.document}:{self.start}-{self.end}"


def _headings(text: str) -> list[tuple[int, tuple[str, ...]]]:
    headings = [(0, ())]
    stack: list[tuple[int, str]] = []
    for match in _HEADING.finditer(text):
        level = len(match[1])
        title = re.sub(r"(?:^|[ \t]+)#+[ \t]*$", "", match[2] or "").strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = tuple(title for _, title in stack)
        if match.start() == 0:
            headings[0] = (0, path)
        else:
            headings.append((match.start(), path))
    return headings


def by_heading(document: str, text: str) -> list[Chunk]:
    headings = _headings(text)
    ends = [offset for offset, _ in headings[1:]] + [len(text)]
    return [
        Chunk(document, path, start, end, text[start:end])
        for (start, path), end in zip(headings, ends)
        if text[start:end].strip()
    ]


def fixed_width(document: str, text: str, size: int, overlap: int) -> list[Chunk]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("需要满足 0 ≤ 重叠字数 < 切块字数")

    headings = _headings(text)
    offsets = [offset for offset, _ in headings]
    chunks = []
    for start in range(0, len(text), size - overlap):
        end = min(start + size, len(text))
        _, path = headings[bisect_right(offsets, start) - 1]
        chunks.append(Chunk(document, path, start, end, text[start:end]))
        if end == len(text):
            break
    return chunks


def gold_chunks(chunks: list[Chunk], gold: str) -> set[str]:
    return {chunk.id for chunk in chunks if gold in chunk.text}
