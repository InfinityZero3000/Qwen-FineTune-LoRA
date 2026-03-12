from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Any

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "DL-Model-Support" / "datasets" / "benchmarks"


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _join_context(record: dict[str, Any], *, max_docs: int = 6, max_sentences_per_doc: int = 4) -> str:
    context = record.get("context") or {}
    titles = _ensure_list(context.get("title"))
    sentences = _ensure_list(context.get("sentences"))

    sections: list[str] = []
    for title, doc_sentences in zip(titles[:max_docs], sentences[:max_docs]):
        snippet = " ".join(str(item).strip() for item in _ensure_list(doc_sentences)[:max_sentences_per_doc] if str(item).strip())
        if snippet:
            sections.append(f"[{title}] {snippet}")
    return "\n".join(sections)


def _extract_context_docs(record: dict[str, Any], *, max_docs: int = 6, max_sentences_per_doc: int = 4) -> list[dict[str, str]]:
    context = record.get("context") or {}
    titles = _ensure_list(context.get("title"))
    sentences = _ensure_list(context.get("sentences"))

    docs: list[dict[str, str]] = []
    for idx, (title, doc_sentences) in enumerate(zip(titles[:max_docs], sentences[:max_docs])):
        snippet = " ".join(
            str(item).strip() for item in _ensure_list(doc_sentences)[:max_sentences_per_doc] if str(item).strip()
        )
        if snippet:
            docs.append({
                "item_id": str(title).strip().lower() or f"doc_{idx}",
                "title": str(title).strip(),
                "text": snippet,
            })
    return docs


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _iter_split_records(dataset_name: str, split_name: str, config_name: str | None = None):
    if config_name:
        return load_dataset(dataset_name, config_name, split=split_name, streaming=True)
    return load_dataset(dataset_name, split=split_name, streaming=True)


def _normalize_hotpot_like(record: dict[str, Any], *, dataset_name: str) -> dict[str, Any]:
    supporting = record.get("supporting_facts") or {}
    supporting_titles = [str(item).strip() for item in _ensure_list(supporting.get("title")) if str(item).strip()]
    return {
        "text": str(record.get("question") or "").strip(),
        "task": "multihop_qa",
        "output": {
            "answer": record.get("answer"),
            "supporting_facts": {
                "title": _ensure_list(supporting.get("title")),
                "sent_id": _ensure_list(supporting.get("sent_id")),
            },
        },
        "metadata": {
            "source_dataset": dataset_name,
            "source_id": record.get("id") or record.get("_id"),
            "question_type": record.get("type"),
            "difficulty": record.get("level"),
            "raw_text": str(record.get("question") or "").strip(),
            "context": _join_context(record),
            "context_docs": _extract_context_docs(record),
            "supporting_titles": sorted(set(supporting_titles)),
        },
    }


def _normalize_ms_marco(record: dict[str, Any], *, dataset_name: str) -> dict[str, Any] | None:
    query = str(record.get("query") or "").strip()
    if not query:
        return None

    answers = [str(item).strip() for item in _ensure_list(record.get("answers")) if str(item).strip()]
    passages = record.get("passages") or {}
    passage_texts = [
        str(item).strip()
        for item in _ensure_list(passages.get("passage_text"))
        if str(item).strip()
    ]
    selected = _ensure_list(passages.get("is_selected"))
    passage_ids = [f"passage_{idx}" for idx in range(len(passage_texts))]

    chosen_context: list[str] = []
    relevant_passage_ids: list[str] = []
    for idx, text in enumerate(passage_texts):
        if idx < len(selected) and selected[idx] == 1:
            chosen_context.append(text)
            relevant_passage_ids.append(passage_ids[idx])
    if not chosen_context:
        chosen_context = passage_texts[:5]

    return {
        "text": query,
        "task": "retrieval_qa",
        "output": {
            "answers": answers,
        },
        "metadata": {
            "source_dataset": dataset_name,
            "source_id": record.get("query_id") or record.get("id"),
            "raw_text": query,
            "well_formed_answer": record.get("wellFormedAnswers"),
            "context": "\n".join(chosen_context),
            "passage_count": len(passage_texts),
            "passages": [
                {
                    "item_id": passage_ids[idx],
                    "title": passage_ids[idx],
                    "text": text,
                    "is_selected": bool(idx < len(selected) and selected[idx] == 1),
                }
                for idx, text in enumerate(passage_texts)
            ],
            "relevant_passage_ids": relevant_passage_ids,
        },
    }


def _normalize_squad(record: dict[str, Any], *, dataset_name: str) -> dict[str, Any] | None:
    query = str(record.get("question") or "").strip()
    context = str(record.get("context") or "").strip()
    answers = record.get("answers") or {}
    answer_texts = [str(item).strip() for item in _ensure_list(answers.get("text")) if str(item).strip()]

    if not query or not context or not answer_texts:
        return None

    source_id = record.get("id") or query[:64]
    return {
        "text": query,
        "task": "extractive_qa",
        "output": {
            "answers": answer_texts,
        },
        "metadata": {
            "source_dataset": dataset_name,
            "source_id": source_id,
            "raw_text": query,
            "context": context,
            "context_docs": [
                {
                    "item_id": str(source_id),
                    "title": str(record.get("title") or source_id),
                    "text": context,
                }
            ],
            "answer_starts": _ensure_list(answers.get("answer_start")),
            "title": record.get("title"),
        },
    }


def _save_split(dataset_root: Path, split_name: str, rows: list[dict[str, Any]]) -> None:
    _write_jsonl(dataset_root / f"{split_name}.jsonl", rows)
    manifest = {
        "split": split_name,
        "count": len(rows),
        "path": str(dataset_root / f"{split_name}.jsonl"),
    }
    (dataset_root / f"{split_name}.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _normalize_and_save(
    dataset_root: Path,
    dataset_name: str,
    split_names: list[str],
    normalizer,
    split_limit: int | None,
    config_name: str | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_name in split_names:
        records = _iter_split_records(dataset_name, split_name, config_name=config_name)
        if split_limit is not None and split_limit > 0:
            records = islice(records, split_limit)
        rows: list[dict[str, Any]] = []
        for record in records:
            normalized = normalizer(record, dataset_name=dataset_name)
            if normalized:
                rows.append(normalized)
        _save_split(dataset_root, split_name, rows)
        counts[split_name] = len(rows)
    return counts


def download_hotpotqa(output_root: Path, split_limit: int | None) -> dict[str, int]:
    dataset_root = output_root / "hotpotqa"
    return _normalize_and_save(
        dataset_root,
        "hotpotqa/hotpot_qa",
        ["train", "validation"],
        _normalize_hotpot_like,
        split_limit,
        config_name="distractor",
    )


def download_2wiki(output_root: Path, split_limit: int | None) -> dict[str, int]:
    dataset_root = output_root / "2wikimultihopqa"
    return _normalize_and_save(
        dataset_root,
        "framolfese/2WikiMultihopQA",
        ["train", "validation", "test"],
        _normalize_hotpot_like,
        split_limit,
    )


def download_ms_marco(output_root: Path, split_limit: int | None) -> dict[str, int]:
    dataset_root = output_root / "ms_marco"
    return _normalize_and_save(
        dataset_root,
        "microsoft/ms_marco",
        ["train", "validation"],
        _normalize_ms_marco,
        split_limit,
        config_name="v1.1",
    )


def download_squad(output_root: Path, split_limit: int | None) -> dict[str, int]:
    dataset_root = output_root / "squad"
    return _normalize_and_save(
        dataset_root,
        "rajpurkar/squad",
        ["train", "validation"],
        _normalize_squad,
        split_limit,
        config_name="plain_text",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize public GraphCAG benchmark datasets.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where normalized benchmark JSONL files will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["hotpotqa", "2wikimultihopqa", "ms_marco", "squad"],
        default=["hotpotqa", "2wikimultihopqa", "ms_marco", "squad"],
        help="Datasets to download and normalize.",
    )
    parser.add_argument(
        "--split-limit",
        type=int,
        default=None,
        help="Optional maximum number of examples to keep per split for faster local experiments.",
    )
    args = parser.parse_args()

    summaries: dict[str, dict[str, int]] = {}
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if "hotpotqa" in args.datasets:
        summaries["hotpotqa"] = download_hotpotqa(output_root, args.split_limit)
    if "2wikimultihopqa" in args.datasets:
        summaries["2wikimultihopqa"] = download_2wiki(output_root, args.split_limit)
    if "ms_marco" in args.datasets:
        summaries["ms_marco"] = download_ms_marco(output_root, args.split_limit)
    if "squad" in args.datasets:
        summaries["squad"] = download_squad(output_root, args.split_limit)

    summary_path = output_root / "manifest.json"
    existing_summaries: dict[str, dict[str, int]] = {}
    if summary_path.exists():
        try:
            existing_payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(existing_payload, dict):
                existing_summaries = {
                    str(name): counts
                    for name, counts in existing_payload.items()
                    if isinstance(name, str) and isinstance(counts, dict)
                }
        except json.JSONDecodeError:
            existing_summaries = {}

    existing_summaries.update(summaries)
    summary_path.write_text(json.dumps(existing_summaries, indent=2), encoding="utf-8")

    print(json.dumps({"output_root": str(output_root), "datasets": existing_summaries}, indent=2))


if __name__ == "__main__":
    main()