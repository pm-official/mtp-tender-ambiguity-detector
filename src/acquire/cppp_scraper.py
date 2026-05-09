"""Selenium-based CPPP semi-automatic downloader.

CPPP gates every search behind a CAPTCHA, so a fully-automatic scraper is
not feasible. The workflow this module supports:

  1. You start the scraper. It opens a real Chrome window via Selenium.
  2. You navigate manually to the tender detail page (search → CAPTCHA →
     pick tender → click "Documents" tab). Solve any CAPTCHAs yourself.
  3. When you're on the tender detail page, press Enter in the terminal.
  4. The scraper extracts every PDF download link from the current page,
     downloads each PDF using the browser's cookies (so server sessions
     line up), validates each file, and writes manifest.json.

Selenium is an optional dependency. The module imports lazily so the rest
of the codebase doesn't require it unless this scraper is actually run.

Usage from CLI:
    python -m src.acquire scrape --tender-id 2024_NHAI_xxx_x --out corpus/

Usage programmatically:
    from src.acquire.cppp_scraper import semi_automatic_download
    folder = semi_automatic_download(out_root, tender_id="...")
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

from src.acquire.manifest import (
    Manifest,
    TenderDocument,
    infer_corrigendum_sequence,
    infer_role_from_filename,
    sha256_of_file,
    write_manifest,
)
from src.acquire.pdf_validator import inspect_pdf, quick_magic_check

logger = logging.getLogger(__name__)


CPPP_HOME = "https://etenders.gov.in/eprocure/app"
CAPTCHA_HINT = (
    "\n  CPPP gates every search behind a CAPTCHA. Use the Chrome window\n"
    "  this script opened to:\n"
    "    1. Navigate to the tender you want.\n"
    "    2. Solve any CAPTCHAs yourself.\n"
    "    3. Open the tender detail page (the page that lists all PDFs).\n"
    "    4. Come back to this terminal and press Enter.\n"
)


def semi_automatic_download(
    out_root: Path,
    *,
    tender_id: str,
    initial_url: str = CPPP_HOME,
    download_dir: Optional[Path] = None,
    headless: bool = False,
) -> Path:
    """Open a Chrome window, wait for the user to land on a tender detail page,
    then download all linked PDFs and write a manifest.

    Returns the corpus folder path.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError as e:
        raise ImportError(
            "Selenium not installed. Run: pip install selenium"
        ) from e

    folder = Path(out_root) / tender_id
    folder.mkdir(parents=True, exist_ok=True)

    print(f"\n[acquire] Opening Chrome — corpus folder: {folder}")
    print(CAPTCHA_HINT)

    options = Options()
    if headless:
        options.add_argument("--headless=new")
    # Make Chrome auto-download to our folder rather than asking the user
    # where to save each PDF. Some CPPP downloads use POST forms; this
    # captures those without us having to re-implement the form submission.
    prefs = {
        "download.default_directory": str(folder.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "plugins.always_open_pdf_externally": True,  # don't render in tab
    }
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(initial_url)

        # Hand off to the user
        input("\n[acquire] Press Enter once you're on the tender detail page... ")

        current_url = driver.current_url
        print(f"\n[acquire] Working from URL:\n  {current_url}")

        # Try to read tender title and authority from the page (best-effort).
        try:
            page_source = driver.page_source
        except Exception as e:
            page_source = ""
            logger.warning("Could not read page source: %s", e)

        title_guess = _guess_tender_title(page_source)
        authority_guess = _guess_issuing_authority(page_source)

        # Find every <a href="...pdf"> on the page and trigger downloads.
        # We click each link with JavaScript so Chrome fires the auto-download
        # via the prefs we set.
        pdf_links = _find_pdf_links(driver)
        if not pdf_links:
            print(
                "\n[acquire] No PDF links detected on this page. Some CPPP "
                "tenders use POST forms or a Documents sub-tab — make sure "
                "you're on the right page. Press Ctrl+C to abort or Enter "
                "to continue with the manifest of whatever is in the folder."
            )
            input()
        else:
            print(f"\n[acquire] Found {len(pdf_links)} PDF links. Triggering downloads...")
            for idx, href in enumerate(pdf_links, 1):
                print(f"  [{idx:2d}/{len(pdf_links)}]  {href}")
                _trigger_download(driver, href)
                time.sleep(2)  # be polite

            print("\n[acquire] Waiting for downloads to finish...")
            _wait_for_downloads(folder, timeout_seconds=120)

        # Build manifest from whatever is in the folder
        manifest = _build_manifest_from_folder(
            folder=folder,
            tender_id=tender_id,
            source_url=current_url,
            issuing_authority=authority_guess,
            project_title=title_guess,
        )
        write_manifest(folder, manifest)
        print(f"\n[acquire] Manifest written to {folder / 'manifest.json'}")
        print(f"[acquire] {manifest.num_documents} documents, "
              f"{manifest.num_corrigenda} corrigenda.")
        return folder
    finally:
        driver.quit()


# ─── Helpers ────────────────────────────────────────────────────────────────
def _find_pdf_links(driver) -> list[str]:
    """Return all unique <a href="*.pdf"> URLs on the current page."""
    from selenium.webdriver.common.by import By

    elements = driver.find_elements(By.CSS_SELECTOR, "a[href]")
    seen: set[str] = set()
    out: list[str] = []
    for el in elements:
        href = el.get_attribute("href") or ""
        if not href:
            continue
        # Match .pdf links (CPPP uses both direct .pdf and download servlets)
        if href.lower().endswith(".pdf") or "DownloadDocument" in href:
            if href not in seen:
                seen.add(href)
                out.append(href)
    return out


def _trigger_download(driver, href: str) -> None:
    """Click a link via JS so Chrome's auto-download kicks in."""
    safe = href.replace("'", "\\'")
    driver.execute_script(
        f"""
        var a = document.createElement('a');
        a.href = '{safe}';
        a.target = '_self';
        document.body.appendChild(a);
        a.click();
        a.remove();
        """
    )


def _wait_for_downloads(folder: Path, *, timeout_seconds: int = 120) -> None:
    """Poll until no Chrome ".crdownload" partials remain in folder."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        partials = list(folder.glob("*.crdownload"))
        if not partials:
            return
        time.sleep(1)
    logger.warning(
        "Downloads still in progress after %ds; you may need to wait "
        "and re-run the validator.", timeout_seconds
    )


def _guess_tender_title(page_source: str) -> str:
    if not page_source:
        return ""
    # CPPP detail page uses several different headings; try a few
    for pattern in (
        r"Tender\s*Title[:\s]*<[^>]*>([^<]+)<",
        r"<title[^>]*>([^<]+)</title>",
    ):
        m = re.search(pattern, page_source, re.IGNORECASE)
        if m:
            return m.group(1).strip()[:200]
    return ""


def _guess_issuing_authority(page_source: str) -> str:
    if not page_source:
        return "UNKNOWN"
    for pattern in (
        r"Organisation\s*Chain[:\s]*<[^>]*>([^<]+)<",
        r"Issuing\s*Authority[:\s]*<[^>]*>([^<]+)<",
    ):
        m = re.search(pattern, page_source, re.IGNORECASE)
        if m:
            return m.group(1).strip()[:200]
    return "UNKNOWN"


def _build_manifest_from_folder(
    *,
    folder: Path,
    tender_id: str,
    source_url: str,
    issuing_authority: str,
    project_title: str,
) -> Manifest:
    documents: list[TenderDocument] = []
    for pdf in sorted(folder.glob("*.pdf")):
        info = inspect_pdf(pdf, sample_text_layer=True)
        role = infer_role_from_filename(pdf.name)
        seq = infer_corrigendum_sequence(pdf.name) if role == "corrigendum" else 0
        documents.append(
            TenderDocument(
                filename=pdf.name,
                role=role,
                sequence=seq,
                page_count=info.page_count,
                file_size_bytes=info.file_size_bytes,
                sha256=sha256_of_file(pdf),
                notes=info.error or "",
            )
        )

    return Manifest(
        tender_id=tender_id,
        source_url=source_url,
        issuing_authority=issuing_authority or "UNKNOWN",
        project_title=project_title or "",
        documents=documents,
        retrieved_at=datetime.now(tz=timezone.utc),
        retrieved_by="cppp_scraper.semi_automatic_download",
        notes=(
            "Acquired via semi-automatic Selenium flow. PDF roles auto-tagged "
            "from filenames; review and edit manifest.json if any role looks "
            "wrong."
        ),
    )
