"""
run_log/report_renderer.py

Генерирует self-contained HTML-отчёт из заполненных полей заявки НВОС.

Публичный интерфейс:
    render_report(result, run_dir, template_path) -> Path

Принимает result в виде dict[field_id -> Pydantic-объект | None]
или dict[field_id -> dict | None] (после json.load).
"""

import html as html_lib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.field_config import FIELD_CONFIG

_FIELD_RE = re.compile(r"<(ПОЛЕ_\d+)>")


# =============================================================================
# CSS (встроенный в HTML — никаких внешних зависимостей)
# =============================================================================

_CSS = """
* { box-sizing: border-box; }
body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 14px;
    line-height: 1.7;
    max-width: 980px;
    margin: 40px auto;
    padding: 0 28px 60px;
    color: #1a1a1a;
    background: #fff;
}
h1 { font-size: 17px; margin: 32px 0 6px; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 15px; margin: 20px 0 4px; }
h3 { font-size: 13px; margin: 14px 0 2px; font-style: italic; font-weight: normal; }
h4 { font-size: 13px; margin: 10px 0 2px; font-weight: normal; color: #555; }
hr  { border: none; border-top: 1px solid #ccc; margin: 12px 0; }
p   { margin: 3px 0; }

/* ── report header ── */
.rh { background: #f0f4ff; border: 1px solid #c5d0e8; border-radius: 6px;
      padding: 12px 18px; margin-bottom: 28px; font-family: sans-serif; font-size: 12px; }
.rh strong { font-size: 15px; display: block; margin-bottom: 4px; }

/* ── inline scalar value ── */
.fv {
    display: inline-block;
    border-radius: 3px;
    padding: 1px 7px;
    font-weight: bold;
    font-family: sans-serif;
    font-size: 13px;
    vertical-align: middle;
    line-height: 1.4;
}
.fv.hi  { background: #d4edda; border: 1px solid #28a745; color: #155724; }
.fv.mid { background: #fff3cd; border: 1px solid #ffc107; color: #856404; }
.fv.lo  { background: #f8d7da; border: 1px solid #dc3545; color: #721c24; }

/* ── not found ── */
.nf {
    display: inline-block;
    color: #721c24; background: #f8d7da;
    border: 1px solid #dc3545; border-radius: 3px;
    padding: 1px 7px;
    font-family: sans-serif; font-size: 12px; font-style: italic;
    vertical-align: middle;
}

/* ── confidence badge ── */
.cb { font-size: 10px; font-family: monospace; opacity: 0.65; margin-left: 3px; vertical-align: super; }

/* ── collapsible source fragment ── */
details.src { display: inline-block; margin-left: 5px; font-size: 11px; font-family: sans-serif; }
details.src summary { cursor: pointer; color: #888; }
details.src div {
    background: #f7f7f7; border-left: 3px solid #bbb;
    padding: 4px 10px; margin-top: 3px;
    font-size: 11px; font-style: italic;
    max-width: 640px; white-space: pre-wrap;
    display: block;
}

/* ── block fields (table / coordinates / requisites) ── */
.fb {
    margin: 6px 0;
    padding: 8px 14px;
    border-radius: 4px;
    border-left: 4px solid #ccc;
    font-family: sans-serif; font-size: 13px;
}
.fb.hi  { border-left-color: #28a745; background: #f6fff8; }
.fb.mid { border-left-color: #ffc107; background: #fffdf0; }
.fb.lo  { border-left-color: #dc3545; background: #fff6f6; }
.fb .fl { font-size: 11px; color: #666; margin-bottom: 5px; }

/* ── HTML table ── */
.ft { border-collapse: collapse; width: 100%; margin-top: 6px; font-size: 12px; }
.ft th { background: #eee; border: 1px solid #bbb; padding: 4px 8px; text-align: left; white-space: nowrap; }
.ft td { border: 1px solid #ddd; padding: 3px 8px; }
.ft tr:nth-child(even) td { background: #fafafa; }

/* ── requisites list ── */
.rl { list-style: none; padding: 0; margin: 4px 0; }
.rl li { margin: 2px 0; }

/* ── coordinates list ── */
.cl { list-style: none; padding: 0; margin: 4px 0; font-family: monospace; font-size: 12px; }
.cl li { margin: 2px 0; }
"""


# =============================================================================
# Helpers
# =============================================================================

def _conf_cls(confidence: float) -> str:
    if confidence >= 0.8:
        return "hi"
    if confidence >= 0.5:
        return "mid"
    return "lo"


def _source_details(source: str) -> str:
    if not source:
        return ""
    return (
        '<details class="src"><summary>источник</summary>'
        f'<div>{html_lib.escape(source)}</div></details>'
    )


def _field_data(value_obj) -> Optional[dict]:
    """Нормализует Pydantic-объект или dict → dict, или None."""
    if value_obj is None:
        return None
    if hasattr(value_obj, "model_dump"):
        return value_obj.model_dump()
    if isinstance(value_obj, dict):
        return value_obj
    return None


# =============================================================================
# Рендереры по типу поля
# =============================================================================

def _not_found(field_id: str) -> str:
    desc = FIELD_CONFIG.get(field_id, {}).get("description", "")
    title = html_lib.escape(f"{field_id}: {desc[:90]}")
    return f'<span class="nf" title="{title}">⚠ {field_id}: не найдено</span>'


def _render_scalar(field_id: str, data: dict, cls: str) -> str:
    value = data.get("value", "")
    if isinstance(value, bool):
        value = "Да" if value else "Нет"
    confidence = data.get("confidence", 0.0)
    source = data.get("source_fragment", "")
    desc = FIELD_CONFIG.get(field_id, {}).get("description", "")
    title = html_lib.escape(f"{field_id}: {desc[:100]}")
    badge = f'<span class="cb">[{confidence:.2f}]</span>'
    return (
        f'<span class="fv {cls}" title="{title}">{html_lib.escape(str(value))}</span>'
        f'{badge}{_source_details(source)}'
    )


def _render_requisites(field_id: str, data: dict, cls: str) -> str:
    confidence = data.get("confidence", 0.0)
    desc = FIELD_CONFIG.get(field_id, {}).get("description", "")
    label = html_lib.escape(f"{field_id}: {desc[:90]}")
    items = []
    mapping = [
        ("document_number",   "Номер"),
        ("issue_date",        "Дата выдачи"),
        ("issuing_authority", "Орган"),
        ("expiry_date",       "Срок действия"),
    ]
    for key, title in mapping:
        val = data.get(key)
        if val:
            items.append(f'<li><b>{title}:</b> {html_lib.escape(str(val))}</li>')
    items_html = "\n".join(items) or "<li><i>нет данных</i></li>"
    return (
        f'<div class="fb {cls}">'
        f'<div class="fl">{label} [{confidence:.2f}]</div>'
        f'<ul class="rl">{items_html}</ul>'
        f'{_source_details(data.get("source_fragment", ""))}</div>'
    )


def _render_table(field_id: str, data: dict, cls: str) -> str:
    rows = data.get("rows", [])
    confidence = data.get("confidence", 0.0)
    desc = FIELD_CONFIG.get(field_id, {}).get("description", "")
    label = html_lib.escape(f"{field_id}: {desc[:90]}")
    if not rows:
        return (
            f'<div class="fb {cls}"><div class="fl">{label} [{confidence:.2f}]</div>'
            f'<i>нет строк</i></div>'
        )
    headers = list(rows[0].keys())
    thead = "<tr>" + "".join(f"<th>{html_lib.escape(h)}</th>" for h in headers) + "</tr>"
    tbody_rows = [
        "<tr>" + "".join(
            f"<td>{html_lib.escape(str(row.get(h, '')))}</td>" for h in headers
        ) + "</tr>"
        for row in rows
    ]
    return (
        f'<div class="fb {cls}">'
        f'<div class="fl">{label} [{confidence:.2f}]</div>'
        f'<table class="ft"><thead>{thead}</thead>'
        f'<tbody>{"".join(tbody_rows)}</tbody></table>'
        f'{_source_details(data.get("source_fragment", ""))}</div>'
    )


def _render_coordinates(field_id: str, data: dict, cls: str) -> str:
    points = data.get("points", [])
    confidence = data.get("confidence", 0.0)
    desc = FIELD_CONFIG.get(field_id, {}).get("description", "")
    label = html_lib.escape(f"{field_id}: {desc[:90]}")
    items = []
    for i, pt in enumerate(points, 1):
        src_id = pt.get("source_id") or f"#{i}"
        lat = pt.get("latitude") or "—"
        lon = pt.get("longitude") or "—"
        cs = pt.get("coordinate_system") or ""
        cs_str = f' ({html_lib.escape(cs)})' if cs else ""
        items.append(
            f"<li>{html_lib.escape(str(src_id))}: "
            f"{html_lib.escape(str(lat))}, {html_lib.escape(str(lon))}"
            f"{cs_str}</li>"
        )
    items_html = "\n".join(items) or "<li><i>нет точек</i></li>"
    return (
        f'<div class="fb {cls}">'
        f'<div class="fl">{label} [{confidence:.2f}]</div>'
        f'<ul class="cl">{items_html}</ul>'
        f'{_source_details(data.get("source_fragment", ""))}</div>'
    )


def _render_field_token(field_id: str, result: dict) -> str:
    """Возвращает HTML-виджет для одного токена <ПОЛЕ_N>."""
    data = _field_data(result.get(field_id))
    if data is None:
        return _not_found(field_id)
    cls = _conf_cls(data.get("confidence", 0.0))
    ext_type = FIELD_CONFIG.get(field_id, {}).get("extraction_type", "short_string")
    if ext_type == "table":
        return _render_table(field_id, data, cls)
    if ext_type == "coordinates":
        return _render_coordinates(field_id, data, cls)
    if ext_type == "requisites":
        return _render_requisites(field_id, data, cls)
    return _render_scalar(field_id, data, cls)


# =============================================================================
# Парсинг шаблона
# =============================================================================

def _substitute_fields(line: str, result: dict) -> str:
    """
    Заменяет все <ПОЛЕ_N> в строке на HTML-виджеты,
    экранируя остальной текст.
    """
    parts = []
    last = 0
    for m in _FIELD_RE.finditer(line):
        parts.append(html_lib.escape(line[last:m.start()]))
        parts.append(_render_field_token(m.group(1), result))
        last = m.end()
    parts.append(html_lib.escape(line[last:]))
    return "".join(parts)


def _process_line(raw: str, result: dict) -> str:
    """Конвертирует одну строку шаблона в HTML."""
    s = raw.strip()

    # ── структурные маркеры <h1> / <h2> / <h3> ──
    if s.startswith("<h1>"):
        content = s[4:].split("//")[0].strip()
        if "конец" in content.lower():
            return "<hr>"
        return f"<h2>{html_lib.escape(content)}</h2>"

    if s.startswith("<h2>"):
        inner = s[4:].strip().strip("<>").strip()
        if "конец" in inner.lower():
            return ""
        inner = inner.split("//")[0].strip().strip("<>").strip()
        return f"<h3>{html_lib.escape(inner)}</h3>"

    if s.startswith("<h3>"):
        inner = s[4:].strip().strip("<>").strip()
        if "конец" in inner.lower():
            return ""
        inner = inner.split("//")[0].strip().strip("<>").strip()
        return f"<h4>{html_lib.escape(inner)}</h4>"

    # ── строки с токенами полей ──
    if _FIELD_RE.search(s):
        return f"<p>{_substitute_fields(s, result)}</p>"

    # ── пустые строки ──
    if not s:
        return ""

    # ── обычный текст ──
    return f"<p>{html_lib.escape(s)}</p>"


# =============================================================================
# HTML-обёртка
# =============================================================================

def _wrap_html(body: str, result: dict, run_dir: Path) -> str:
    found = sum(1 for v in result.values() if _field_data(v) is not None)
    total = len(result)
    now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Заявка НВОС — отчёт</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="rh">
  <strong>Заявка НВОС — автоматически заполненный отчёт</strong>
  Директория рана: <code>{html_lib.escape(str(run_dir))}</code><br>
  Сгенерирован: {now} &nbsp;·&nbsp;
  Заполнено полей: <b>{found}</b> из <b>{total}</b>
  (не найдено: <b>{total - found}</b>)
</div>
{body}
</body>
</html>"""


# =============================================================================
# Публичный API
# =============================================================================

def render_report(result: dict, run_dir: Path, template_path: Path) -> Path:
    """
    Генерирует report.html в run_dir.

    Args:
        result:        dict[field_id -> Pydantic-объект | dict | None]
        run_dir:       директория текущего рана (рядом с result.json)
        template_path: путь к nvoc_txt.txt

    Returns:
        Path к сгенерированному report.html
    """
    template_lines = template_path.read_text(encoding="utf-8").splitlines()
    html_lines = [_process_line(line, result) for line in template_lines]
    body = "\n".join(html_lines)

    html_content = _wrap_html(body, result, run_dir)
    output_path = run_dir / "report.html"
    output_path.write_text(html_content, encoding="utf-8")
    return output_path
