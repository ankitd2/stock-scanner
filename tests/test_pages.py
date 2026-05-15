"""
tests/test_pages.py — Unit tests for output/pages.py

Tests cover:
  - stage_report creates files at the correct paths
  - update_index generates valid HTML containing report links
  - get_all_report_dates parses filenames correctly
  - cleanup_old_reports removes the correct files
"""

import sys
import importlib
from pathlib import Path
from datetime import date

import pytest

# ---------------------------------------------------------------------------
# Fixture: redirect SITE_DIR to a temporary directory
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_site(tmp_path, monkeypatch):
    """
    Patch pages.SITE_DIR to point at a fresh temp directory so tests are
    isolated and never touch the real site/ folder.
    """
    import output.pages as pages
    monkeypatch.setattr(pages, "SITE_DIR", tmp_path / "site")
    return tmp_path / "site"


# ---------------------------------------------------------------------------
# stage_report
# ---------------------------------------------------------------------------

class TestStageReport:
    def test_creates_dated_file(self, tmp_site):
        import output.pages as pages
        d = date(2026, 5, 12)
        result = pages.stage_report("<html>weekly</html>", "weekly", d)
        assert result == tmp_site / "weekly" / "2026-05-12.html"
        assert result.exists()
        assert result.read_text(encoding="utf-8") == "<html>weekly</html>"

    def test_creates_latest_file(self, tmp_site):
        import output.pages as pages
        d = date(2026, 5, 12)
        pages.stage_report("<html>latest</html>", "weekly", d)
        latest = tmp_site / "latest_weekly.html"
        assert latest.exists()
        assert latest.read_text(encoding="utf-8") == "<html>latest</html>"

    def test_creates_directories(self, tmp_site):
        import output.pages as pages
        assert not (tmp_site / "daily").exists()
        pages.stage_report("<html/>", "daily", date(2026, 5, 1))
        assert (tmp_site / "daily").is_dir()

    def test_defaults_to_today(self, tmp_site):
        import output.pages as pages
        result = pages.stage_report("<html/>", "daily")
        assert result.name == f"{date.today().isoformat()}.html"
        assert result.exists()

    def test_overwrites_existing(self, tmp_site):
        import output.pages as pages
        d = date(2026, 5, 12)
        pages.stage_report("<html>v1</html>", "weekly", d)
        pages.stage_report("<html>v2</html>", "weekly", d)
        assert (tmp_site / "weekly" / "2026-05-12.html").read_text() == "<html>v2</html>"

    def test_daily_and_weekly_independent(self, tmp_site):
        import output.pages as pages
        d = date(2026, 5, 12)
        pages.stage_report("<html>daily</html>", "daily", d)
        pages.stage_report("<html>weekly</html>", "weekly", d)
        assert (tmp_site / "daily" / "2026-05-12.html").read_text() == "<html>daily</html>"
        assert (tmp_site / "weekly" / "2026-05-12.html").read_text() == "<html>weekly</html>"
        assert (tmp_site / "latest_daily.html").read_text() == "<html>daily</html>"
        assert (tmp_site / "latest_weekly.html").read_text() == "<html>weekly</html>"

    def test_returns_path_object(self, tmp_site):
        import output.pages as pages
        result = pages.stage_report("<html/>", "weekly", date(2026, 1, 1))
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# get_all_report_dates
# ---------------------------------------------------------------------------

class TestGetAllReportDates:
    def _seed(self, tmp_site, rtype, dates):
        d = tmp_site / rtype
        d.mkdir(parents=True, exist_ok=True)
        for dt in dates:
            (d / f"{dt.isoformat()}.html").write_text("<html/>")

    def test_returns_sorted_newest_first(self, tmp_site):
        import output.pages as pages
        input_dates = [date(2026, 5, 1), date(2026, 5, 12), date(2026, 4, 28)]
        self._seed(tmp_site, "weekly", input_dates)
        result = pages.get_all_report_dates("weekly")
        assert result == sorted(input_dates, reverse=True)

    def test_empty_when_directory_missing(self, tmp_site):
        import output.pages as pages
        result = pages.get_all_report_dates("weekly")
        assert result == []

    def test_empty_when_no_matching_files(self, tmp_site):
        import output.pages as pages
        d = tmp_site / "weekly"
        d.mkdir(parents=True)
        (d / "README.txt").write_text("ignore me")
        (d / "latest.html").write_text("ignore me too")
        result = pages.get_all_report_dates("weekly")
        assert result == []

    def test_ignores_malformed_filenames(self, tmp_site):
        import output.pages as pages
        d = tmp_site / "weekly"
        d.mkdir(parents=True)
        (d / "2026-05-12.html").write_text("<html/>")
        (d / "not-a-date.html").write_text("<html/>")
        (d / "2026-13-99.html").write_text("<html/>")  # invalid date
        result = pages.get_all_report_dates("weekly")
        assert result == [date(2026, 5, 12)]

    def test_returns_date_objects(self, tmp_site):
        import output.pages as pages
        self._seed(tmp_site, "daily", [date(2026, 5, 10)])
        result = pages.get_all_report_dates("daily")
        assert all(isinstance(d, date) for d in result)

    def test_daily_and_weekly_independent(self, tmp_site):
        import output.pages as pages
        self._seed(tmp_site, "daily", [date(2026, 5, 14)])
        self._seed(tmp_site, "weekly", [date(2026, 5, 12)])
        assert pages.get_all_report_dates("daily") == [date(2026, 5, 14)]
        assert pages.get_all_report_dates("weekly") == [date(2026, 5, 12)]


# ---------------------------------------------------------------------------
# update_index
# ---------------------------------------------------------------------------

class TestUpdateIndex:
    def _seed(self, tmp_site, rtype, dates):
        d = tmp_site / rtype
        d.mkdir(parents=True, exist_ok=True)
        for dt in dates:
            (d / f"{dt.isoformat()}.html").write_text("<html/>")

    def test_creates_index_html(self, tmp_site):
        import output.pages as pages
        pages.update_index("weekly", date(2026, 5, 12))
        assert (tmp_site / "index.html").exists()

    def test_index_contains_report_links(self, tmp_site):
        import output.pages as pages
        self._seed(tmp_site, "weekly", [date(2026, 5, 12), date(2026, 5, 5)])
        pages.update_index("weekly", date(2026, 5, 12))
        content = (tmp_site / "index.html").read_text()
        assert 'href="weekly/2026-05-12.html"' in content
        assert 'href="weekly/2026-05-05.html"' in content

    def test_index_contains_daily_links(self, tmp_site):
        import output.pages as pages
        self._seed(tmp_site, "daily", [date(2026, 5, 14), date(2026, 5, 13)])
        pages.update_index("daily", date(2026, 5, 14))
        content = (tmp_site / "index.html").read_text()
        assert 'href="daily/2026-05-14.html"' in content
        assert 'href="daily/2026-05-13.html"' in content

    def test_index_has_dark_theme(self, tmp_site):
        import output.pages as pages
        pages.update_index("weekly", date(2026, 5, 12))
        content = (tmp_site / "index.html").read_text()
        assert "#0d1117" in content
        assert "#e6edf3" in content
        assert "#58a6ff" in content

    def test_index_has_title(self, tmp_site):
        import output.pages as pages
        pages.update_index("weekly", date(2026, 5, 12))
        content = (tmp_site / "index.html").read_text()
        assert "Market Intelligence Scanner" in content

    def test_index_has_disclaimer(self, tmp_site):
        import output.pages as pages
        pages.update_index("weekly", date(2026, 5, 12))
        content = (tmp_site / "index.html").read_text()
        assert "Not financial advice" in content

    def test_index_shows_last_run_date(self, tmp_site):
        import output.pages as pages
        pages.update_index("weekly", date(2026, 5, 12))
        content = (tmp_site / "index.html").read_text()
        assert "May 12, 2026" in content

    def test_index_caps_at_30_entries(self, tmp_site):
        import output.pages as pages
        # Seed 35 weekly reports
        dates = [date(2026, 1, 1).replace(day=1)]
        from datetime import timedelta
        start = date(2025, 6, 1)
        all_dates = [start + timedelta(weeks=i) for i in range(35)]
        self._seed(tmp_site, "weekly", all_dates)
        pages.update_index("weekly")
        content = (tmp_site / "index.html").read_text()
        # Count weekly hrefs
        import re
        links = re.findall(r'href="weekly/\d{4}-\d{2}-\d{2}\.html"', content)
        assert len(links) <= 30

    def test_index_does_not_fail_without_directories(self, tmp_site):
        import output.pages as pages
        # Neither site/daily nor site/weekly exist
        pages.update_index("weekly")  # should not raise
        assert (tmp_site / "index.html").exists()

    def test_index_newest_first(self, tmp_site):
        import output.pages as pages
        self._seed(tmp_site, "weekly", [date(2026, 4, 1), date(2026, 5, 12)])
        pages.update_index("weekly")
        content = (tmp_site / "index.html").read_text()
        pos_may = content.find("2026-05-12")
        pos_apr = content.find("2026-04-01")
        assert pos_may < pos_apr, "Newer date should appear before older date"

    def test_index_valid_html_structure(self, tmp_site):
        import output.pages as pages
        pages.update_index("weekly")
        content = (tmp_site / "index.html").read_text()
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content
        assert "<head>" in content
        assert "<body>" in content

    def test_index_both_sections_present(self, tmp_site):
        import output.pages as pages
        pages.update_index("weekly")
        content = (tmp_site / "index.html").read_text()
        assert "Weekly Reports" in content
        assert "Daily Reports" in content


# ---------------------------------------------------------------------------
# cleanup_old_reports
# ---------------------------------------------------------------------------

class TestCleanupOldReports:
    def _seed(self, tmp_site, rtype, dates):
        d = tmp_site / rtype
        d.mkdir(parents=True, exist_ok=True)
        for dt in dates:
            (d / f"{dt.isoformat()}.html").write_text("<html/>")

    def test_removes_oldest_beyond_keep(self, tmp_site):
        import output.pages as pages
        from datetime import timedelta
        start = date(2026, 1, 1)
        dates = [start + timedelta(days=i) for i in range(35)]
        self._seed(tmp_site, "weekly", dates)
        deleted = pages.cleanup_old_reports("weekly", keep=30)
        assert deleted == 5
        remaining = list((tmp_site / "weekly").iterdir())
        assert len(remaining) == 30

    def test_keeps_newest_files(self, tmp_site):
        import output.pages as pages
        from datetime import timedelta
        start = date(2026, 1, 1)
        dates = sorted([start + timedelta(days=i) for i in range(35)], reverse=True)
        self._seed(tmp_site, "weekly", dates)
        pages.cleanup_old_reports("weekly", keep=30)
        # The 30 newest should still exist
        for d in dates[:30]:
            assert (tmp_site / "weekly" / f"{d.isoformat()}.html").exists()
        # The 5 oldest should be gone
        for d in dates[30:]:
            assert not (tmp_site / "weekly" / f"{d.isoformat()}.html").exists()

    def test_returns_zero_when_nothing_to_delete(self, tmp_site):
        import output.pages as pages
        from datetime import timedelta
        start = date(2026, 1, 1)
        dates = [start + timedelta(days=i) for i in range(10)]
        self._seed(tmp_site, "weekly", dates)
        deleted = pages.cleanup_old_reports("weekly", keep=30)
        assert deleted == 0

    def test_returns_zero_when_directory_missing(self, tmp_site):
        import output.pages as pages
        deleted = pages.cleanup_old_reports("weekly", keep=30)
        assert deleted == 0

    def test_returns_count_of_deleted(self, tmp_site):
        import output.pages as pages
        from datetime import timedelta
        start = date(2026, 3, 1)
        dates = [start + timedelta(days=i) for i in range(10)]
        self._seed(tmp_site, "daily", dates)
        deleted = pages.cleanup_old_reports("daily", keep=7)
        assert deleted == 3

    def test_exact_keep_boundary(self, tmp_site):
        import output.pages as pages
        from datetime import timedelta
        start = date(2026, 1, 1)
        dates = [start + timedelta(days=i) for i in range(30)]
        self._seed(tmp_site, "weekly", dates)
        deleted = pages.cleanup_old_reports("weekly", keep=30)
        assert deleted == 0
        assert len(list((tmp_site / "weekly").iterdir())) == 30
