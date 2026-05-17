"""
Dataset download and loading utilities.

The assignment uses Project Gutenberg's plain-text copy of The Adventures of
Sherlock Holmes. This module keeps the download automatic, validates the file,
and strips Gutenberg boilerplate before preprocessing.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from .config import DEFAULT_CONFIG, DataConfig
except ImportError:  # pragma: no cover - supports direct script execution
    from config import DEFAULT_CONFIG, DataConfig


LOGGER = logging.getLogger(__name__)
DATA_CONFIG = DEFAULT_CONFIG.data
DATA_DIR = DATA_CONFIG.data_dir
FILE_PATH = DATA_CONFIG.corpus_path
CORPUS_URLS = DATA_CONFIG.corpus_urls


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a concise console logger when scripts are run directly."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _make_session(retries: int) -> requests.Session:
    """Create a requests session with retry logic for transient server errors."""
    retry_policy = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry_policy)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "next-word-prediction-assignment/1.0 "
                "(educational Project Gutenberg corpus download)"
            )
        }
    )
    return session


def _looks_like_sherlock(text: str, min_chars: int) -> bool:
    """Return True when downloaded text is plausibly the expected corpus."""
    if len(text) < min_chars:
        return False
    lowered = text.lower()
    return "sherlock holmes" in lowered and "arthur conan doyle" in lowered


def download_corpus(
    save_path: str | Path = FILE_PATH,
    urls: Sequence[str] = CORPUS_URLS,
    *,
    force: bool = False,
    config: DataConfig = DATA_CONFIG,
) -> str:
    """
    Ensure the Sherlock Holmes corpus exists locally.

    Args:
        save_path: Destination text file path.
        urls: Candidate Project Gutenberg URLs and mirrors.
        force: Re-download even when a valid local file already exists.
        config: Download validation and retry settings.

    Returns:
        The local corpus path as a string for backward compatibility.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and not force:
        existing_text = save_path.read_text(encoding="utf-8", errors="replace")
        if _looks_like_sherlock(existing_text, config.min_corpus_chars):
            LOGGER.info("Corpus already available at %s", save_path)
            return str(save_path)
        LOGGER.warning("Existing corpus at %s looks incomplete; re-downloading.", save_path)

    session = _make_session(retries=config.retries_per_url)
    last_error: Exception | None = None

    for url in urls:
        for attempt in range(1, config.retries_per_url + 1):
            try:
                LOGGER.info("Downloading corpus from %s (attempt %s)", url, attempt)
                response = session.get(url, timeout=config.request_timeout_seconds)
                response.raise_for_status()
                text = response.text
                if not _looks_like_sherlock(text, config.min_corpus_chars):
                    raise ValueError(
                        "Downloaded file did not match the expected Sherlock Holmes corpus"
                    )

                tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
                tmp_path.write_text(text, encoding="utf-8")
                tmp_path.replace(save_path)
                LOGGER.info("Corpus saved to %s (%s characters)", save_path, len(text))
                return str(save_path)
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                LOGGER.warning("Download failed from %s: %s", url, exc)
                if attempt < config.retries_per_url:
                    time.sleep(min(2**attempt, 10))

    raise RuntimeError(
        "Could not download The Adventures of Sherlock Holmes from any configured "
        f"source. Place the UTF-8 plain text file at {save_path}. "
        "Primary source: https://www.gutenberg.org/files/1661/1661-0.txt"
    ) from last_error


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Remove Project Gutenberg legal header/footer while keeping story text.

    Handles merged multi-book files by extracting ALL story sections between
    every START/END marker pair and joining them.
    """
    start_pattern = re.compile(
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        flags=re.IGNORECASE | re.DOTALL,
    )
    end_pattern = re.compile(
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        flags=re.IGNORECASE | re.DOTALL,
    )

    starts = list(start_pattern.finditer(raw_text))
    ends = list(end_pattern.finditer(raw_text))

    sections = []
    for start_m, end_m in zip(starts, ends):
        if start_m.end() < end_m.start():
            sections.append(raw_text[start_m.end() : end_m.start()].strip())

    if sections:
        LOGGER.info("Extracted %d story section(s) from corpus.", len(sections))
        return "\n\n".join(sections)

    LOGGER.warning("Gutenberg markers not found; using full corpus text.")
    return raw_text.strip()


def load_corpus(
    file_path: str | Path = FILE_PATH,
    *,
    download_if_missing: bool = True,
) -> str:
    """
    Load story text from disk, downloading it first when needed.

    Args:
        file_path: Local corpus path.
        download_if_missing: Automatically call download_corpus when absent.

    Returns:
        The story text without Gutenberg boilerplate.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"Corpus not found at {file_path}")
        download_corpus(file_path)

    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    story_text = strip_gutenberg_boilerplate(raw_text)
    LOGGER.info(
        "Corpus loaded: %s characters, approximately %s words",
        f"{len(story_text):,}",
        f"{len(story_text.split()):,}",
    )
    return story_text


def corpus_statistics(text: str) -> dict[str, int]:
    """Return simple corpus statistics used by the notebook and README."""
    words = text.split()
    return {
        "characters": len(text),
        "words": len(words),
        "unique_case_insensitive_words": len({word.lower() for word in words}),
    }


if __name__ == "__main__":
    configure_logging()
    download_corpus()
    corpus = load_corpus()
    print(corpus[:500])
