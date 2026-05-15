"""
output/pages.py — GitHub Pages site staging module.

Stages the site/ directory for deployment to GitHub Pages.
Manages dated archives and a rolling index changelog.
"""

import sys
import os
import re
from pathlib import Path
from datetime import date

# Allow importing from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

SITE_DIR = Path(__file__).parent.parent / "site"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def stage_report(
    html_content: str,
    report_type: str,
    report_date: date = None,
) -> Path:
    """
    Write html_content to:
      site/{report_type}/YYYY-MM-DD.html   (dated archive)
      site/latest_{report_type}.html        (stable "latest" URL)

    Creates directories if needed.
    Returns the path of the dated file.
    """
    if report_date is None:
        report_date = date.today()

    try:
        archive_dir = SITE_DIR / report_type
        archive_dir.mkdir(parents=True, exist_ok=True)

        dated_path = archive_dir / f"{report_date.isoformat()}.html"
        dated_path.write_text(html_content, encoding="utf-8")

        latest_path = SITE_DIR / f"latest_{report_type}.html"
        latest_path.write_text(html_content, encoding="utf-8")

        return dated_path

    except Exception:
        import traceback
        print(f"[pages] ERROR in stage_report: {traceback.format_exc()}", file=sys.stderr)
        return SITE_DIR / report_type / f"{(report_date or date.today()).isoformat()}.html"


def update_index(report_type: str, report_date: date = None) -> None:
    """
    Regenerates site/index.html with a changelog listing all available reports.

    Format: dark-themed HTML page listing:
      - Title: "Market Intelligence Scanner"
      - Subtitle: "Automated daily + weekly market brief"
      - Two sections: "Weekly Reports" and "Daily Reports"
      - Each entry: date (as link to the dated file) + day of week
      - Newest first
      - Maximum 30 entries per section
    """
    try:
        SITE_DIR.mkdir(parents=True, exist_ok=True)

        weekly_dates = get_all_report_dates("weekly")
        daily_dates = get_all_report_dates("daily")

        last_run = report_date or date.today()

        def _build_rows(dates, rtype, label):
            rows = []
            for d in dates[:30]:
                dow = d.strftime("%a")
                friendly = d.strftime("%b %d, %Y")
                href = f"{rtype}/{d.isoformat()}.html"
                rows.append(
                    f'        <li>'
                    f'<a href="{href}">{dow}, {friendly} — {label}</a>'
                    f'</li>'
                )
            return "\n".join(rows) if rows else "        <li>No reports yet.</li>"

        weekly_rows = _build_rows(weekly_dates, "weekly", "Weekly Brief")
        daily_rows = _build_rows(daily_dates, "daily", "Daily Brief")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Market Intelligence Scanner</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 16px;
    line-height: 1.6;
    padding: 2rem 1rem;
  }}
  .container {{
    max-width: 760px;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    color: #e6edf3;
    margin-bottom: 0.25rem;
  }}
  .subtitle {{
    color: #8b949e;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
  }}
  .last-run {{
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 0.35rem 0.75rem;
    font-size: 0.85rem;
    color: #8b949e;
    margin-bottom: 2rem;
  }}
  .last-run span {{
    color: #58a6ff;
    font-weight: 600;
  }}
  section {{
    margin-bottom: 2.5rem;
  }}
  h2 {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #e6edf3;
    border-bottom: 1px solid #21262d;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
  }}
  ul {{
    list-style: none;
  }}
  li {{
    padding: 0.4rem 0;
    border-bottom: 1px solid #161b22;
  }}
  li:last-child {{
    border-bottom: none;
  }}
  a {{
    color: #58a6ff;
    text-decoration: none;
    font-size: 0.95rem;
  }}
  a:hover {{
    text-decoration: underline;
    color: #79c0ff;
  }}
  footer {{
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #21262d;
    color: #6e7681;
    font-size: 0.8rem;
    text-align: center;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Market Intelligence Scanner</h1>
  <p class="subtitle">Automated daily + weekly market brief</p>
  <div class="last-run">Last run: <span>{last_run.strftime("%A, %B %d, %Y")}</span></div>

  <section>
    <h2>Weekly Reports</h2>
    <ul>
{weekly_rows}
    </ul>
  </section>

  <section>
    <h2>Daily Reports</h2>
    <ul>
{daily_rows}
    </ul>
  </section>

  <footer>
    Reports auto-generated by GitHub Actions. Not financial advice.
  </footer>
</div>
</body>
</html>
"""

        index_path = SITE_DIR / "index.html"
        index_path.write_text(html, encoding="utf-8")

    except Exception:
        import traceback
        print(f"[pages] ERROR in update_index: {traceback.format_exc()}", file=sys.stderr)


def get_all_report_dates(report_type: str) -> list:
    """
    Scan site/{report_type}/ for YYYY-MM-DD.html files.
    Returns sorted list of dates, newest first.
    """
    results = []
    try:
        report_dir = SITE_DIR / report_type
        if not report_dir.exists():
            return results

        pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})\.html$")
        for p in report_dir.iterdir():
            m = pattern.match(p.name)
            if m:
                try:
                    d = date.fromisoformat(m.group(1))
                    results.append(d)
                except ValueError:
                    pass  # skip malformed dates

        results.sort(reverse=True)
    except Exception:
        import traceback
        print(f"[pages] ERROR in get_all_report_dates: {traceback.format_exc()}", file=sys.stderr)

    return results


def cleanup_old_reports(report_type: str, keep: int = 30) -> int:
    """
    Delete report HTML files older than the most recent `keep` entries.
    Returns count of deleted files.
    """
    deleted = 0
    try:
        all_dates = get_all_report_dates(report_type)
        to_delete = all_dates[keep:]  # already sorted newest-first, so tail is oldest

        report_dir = SITE_DIR / report_type
        for d in to_delete:
            path = report_dir / f"{d.isoformat()}.html"
            try:
                path.unlink()
                deleted += 1
            except Exception:
                import traceback
                print(
                    f"[pages] WARNING: could not delete {path}: {traceback.format_exc()}",
                    file=sys.stderr,
                )
    except Exception:
        import traceback
        print(f"[pages] ERROR in cleanup_old_reports: {traceback.format_exc()}", file=sys.stderr)

    return deleted
