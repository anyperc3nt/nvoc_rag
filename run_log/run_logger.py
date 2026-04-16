import json
import subprocess
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


def _get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def _make_run_dir(model_name: str, base_dir: str = "logs") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    commit = _get_git_commit()

    safe_model = model_name.replace("/", "_").replace(":", "_")
    folder_name = f"{timestamp}__{safe_model}"
    if commit:
        folder_name += f"__{commit}"

    run_dir = Path(base_dir) / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@dataclass
class FieldRetrievalLog:
    field_id: str
    rag_query: str
    retrieved_chunks: list[dict]   # [{"text": ..., "metadata": ..., "score": ...}]


@dataclass
class GroupCallLog:
    group_id: str
    group_name: str
    fields_in_group: list[str]
    retrieval_logs: list[FieldRetrievalLog]
    system_prompt: str
    full_user_prompt: str          # весь контекст, переданный в LLM
    llm_response_raw: str          # сырой JSON-ответ LLM
    llm_response_parsed: dict      # распаршенный (из Pydantic в dict)
    duration_seconds: float


@dataclass
class RunLog:
    model_name: str
    embed_model_name: str
    grouping_strategy: str         # ACTIVE_GROUPING из group_config
    git_commit: Optional[str]
    started_at: str
    finished_at: Optional[str] = None
    group_logs: list[GroupCallLog] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)  # NotFound-поля


class RunLogger:
    def __init__(
        self,
        model_name: str,
        embed_model_name: str,
        grouping_strategy: str,
        base_dir: str = "logs",
    ):
        self.run_dir = _make_run_dir(model_name, base_dir)
        self.run_log = RunLog(
            model_name=model_name,
            embed_model_name=embed_model_name,
            grouping_strategy=grouping_strategy,
            git_commit=_get_git_commit(),
            started_at=datetime.now().isoformat(),
        )
        print(f"[Logger] Run dir: {self.run_dir}")

    def log_group(self, group_log: GroupCallLog) -> None:
        self.run_log.group_logs.append(group_log)

        # Пишем каждую группу сразу — удобно при падении на полпути
        group_path = self.run_dir / f"{group_log.group_id}.json"
        with open(group_path, "w", encoding="utf-8") as f:
            json.dump(asdict(group_log), f, ensure_ascii=False, indent=2)

    def log_issue(self, field_id: str, reason: str) -> None:
        self.run_log.issues.append({"field": field_id, "reason": reason})

    def finalize(self) -> Path:
        self.run_log.finished_at = datetime.now().isoformat()
        summary_path = self.run_dir / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.run_log), f, ensure_ascii=False, indent=2)
        print(f"[Logger] Run complete. Summary: {summary_path}")
        return summary_path

    def write_report(self, result: dict, template_path: Path) -> Path:
        """Генерирует HTML-отчёт рядом с остальными артефактами рана."""
        from run_log.report_renderer import render_report
        report_path = render_report(result, self.run_dir, template_path)
        print(f"[Logger] HTML-отчёт: {report_path}")
        return report_path
