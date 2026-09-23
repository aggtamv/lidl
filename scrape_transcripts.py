"""
Scrape all video transcripts from a YouTube channel.

Requires (install into your own venv):
    pip install yt-dlp youtube-transcript-api

Usage:
    python scrape_transcripts.py

Output (under ./transcripts/):
    video_index.json    - list of all videos found on the channel
    <video_id>.json     - full transcript with timestamps, per video
    <video_id>.txt      - plain-text transcript, per video
    transcripts.jsonl   - one line per successfully fetched transcript
    failures.csv        - videos with no transcript / errors

Notes:
    * Transcripts are preferred in Greek ('el') then English ('en'),
      including auto-generated captions. Adjust PREFERRED_LANGS below.
    * The script is resumable: rerun it and it skips videos already saved.
    * If you hit "CERTIFICATE_VERIFY_FAILED", see the SSL note at the bottom.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import yt_dlp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHANNEL_URL = "https://www.youtube.com/@lidlhellas/videos"
OUT_DIR = Path("transcripts")
PREFERRED_LANGS = ["el", "en"]   # try Greek first, then English
SLEEP_BETWEEN = 1.0              # seconds between transcript requests (be polite)
MAX_RETRIES = 3                  # per-video retries on transient errors
RETRY_BACKOFF = 5.0              # seconds, multiplied by attempt number
WITH_METADATA = True             # fetch description / likes / views / date per video
WITH_COMMENTS = False            # also fetch comments (MUCH slower - many extra requests)


# ---------------------------------------------------------------------------
# Step 1 - enumerate every video on the channel (no per-video download)
# ---------------------------------------------------------------------------
def list_channel_videos(channel_url: str) -> list[dict]:
    opts = {
        "quiet": True,
        "extract_flat": True,   # only metadata, don't resolve each video
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    entries = info.get("entries") or []
    videos = [
        {
            "id": e.get("id"),
            "title": e.get("title"),
            "url": e.get("url") or f"https://www.youtube.com/watch?v={e.get('id')}",
            "duration": e.get("duration"),
        }
        for e in entries
        if e.get("id")
    ]
    return videos


# ---------------------------------------------------------------------------
# Step 1b - per-video metadata (description, likes, views, comments)
# ---------------------------------------------------------------------------
def fetch_metadata(video_url: str, with_comments: bool = False) -> dict:
    """Resolve full metadata for one video. Set with_comments=True to also
    pull comments (much slower - many extra requests per video)."""
    opts = {
        "quiet": True,
        "skip_download": True,
        "getcomments": with_comments,   # pulls the comment threads
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=False)

    comments = None
    if with_comments:
        comments = [
            {
                "author": c.get("author"),
                "text": c.get("text"),
                "likes": c.get("like_count"),
                "time": c.get("timestamp"),
                "parent": c.get("parent"),   # 'root' or parent comment id (replies)
            }
            for c in (info.get("comments") or [])
        ]

    return {
        "description": info.get("description"),
        "like_count": info.get("like_count"),      # may be None if hidden
        "view_count": info.get("view_count"),
        "upload_date": info.get("upload_date"),     # 'YYYYMMDD'
        "duration": info.get("duration"),
        "tags": info.get("tags"),
        "comment_count": info.get("comment_count"),
        "comments": comments,                       # None unless requested
        # 'dislike_count' intentionally omitted - YouTube no longer exposes it
    }


# ---------------------------------------------------------------------------
# Step 2 - transcript fetch, tolerant of both API versions
# ---------------------------------------------------------------------------
def _make_fetcher():
    """Return fetch(video_id) -> list[{'text','start','duration'}] | raise."""
    from youtube_transcript_api import YouTubeTranscriptApi

    # New API (>= 1.0): instance methods .fetch() / .list()
    if hasattr(YouTubeTranscriptApi, "fetch") or not hasattr(
        YouTubeTranscriptApi, "get_transcript"
    ):
        api = YouTubeTranscriptApi()

        def fetch(video_id: str) -> list[dict]:
            fetched = api.fetch(video_id, languages=PREFERRED_LANGS)
            # FetchedTranscript -> list of dicts
            return fetched.to_raw_data()

        return fetch

    # Old API (< 1.0): classmethods
    def fetch(video_id: str) -> list[dict]:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=PREFERRED_LANGS)

    return fetch


def fetch_with_retries(fetch, video_id: str) -> list[dict]:
    from youtube_transcript_api._errors import (
        TranscriptsDisabled,
        NoTranscriptFound,
    )

    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch(video_id)
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            # Permanent for this video - no point retrying.
            raise e
        except Exception as e:  # transient (network / rate limit / parsing)
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    raise last_err  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Step 3 - flatten every saved record into a single CSV (one row per video)
# ---------------------------------------------------------------------------
def write_csv(out_dir: Path) -> int:
    """Read all <video_id>.json records and emit videos.csv (one row each).
    Works standalone on a resumed run - it just reads whatever is on disk."""
    files = sorted(p for p in out_dir.glob("*.json") if p.stem != "video_index")
    cols = [
        "id", "title", "url", "upload_date", "duration",
        "view_count", "like_count", "comment_count",
        "num_segments", "tags", "description", "transcript",
    ]
    csv_path = out_dir / "videos.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for p in files:
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            tags = rec.get("tags") or []
            writer.writerow({
                "id": rec.get("id"),
                "title": rec.get("title"),
                "url": rec.get("url"),
                "upload_date": rec.get("upload_date"),
                "duration": rec.get("duration"),
                "view_count": rec.get("view_count"),
                "like_count": rec.get("like_count"),
                "comment_count": rec.get("comment_count"),
                "num_segments": len(rec.get("segments") or []),
                "tags": "|".join(tags) if isinstance(tags, list) else tags,
                "description": rec.get("description"),
                "transcript": rec.get("text"),
            })
    return len(files)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Enumerating videos from {CHANNEL_URL} ...")
    videos = list_channel_videos(CHANNEL_URL)
    (OUT_DIR / "video_index.json").write_text(
        json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Found {len(videos)} videos. Index saved to video_index.json")

    fetch = _make_fetcher()

    jsonl_path = OUT_DIR / "transcripts.jsonl"
    fail_path = OUT_DIR / "failures.csv"

    ok = skipped = failed = 0
    with jsonl_path.open("a", encoding="utf-8") as jsonl, fail_path.open(
        "a", newline="", encoding="utf-8"
    ) as failf:
        fail_writer = csv.writer(failf)
        if failf.tell() == 0:
            fail_writer.writerow(["video_id", "title", "reason"])

        for i, v in enumerate(videos, 1):
            vid = v["id"]
            json_file = OUT_DIR / f"{vid}.json"
            if json_file.exists():
                skipped += 1
                continue

            # Metadata (description/likes/views/comments) - independent of the
            # transcript, so a caption-less video still gets its data saved.
            meta: dict = {}
            if WITH_METADATA:
                try:
                    meta = fetch_metadata(v["url"], with_comments=WITH_COMMENTS)
                except Exception as e:
                    print(f"[{i}/{len(videos)}] meta WARN {vid} - {type(e).__name__}: {e}")

            try:
                segments = fetch_with_retries(fetch, vid)
            except Exception as e:
                # No transcript, but still persist whatever metadata we have.
                reason = f"{type(e).__name__}: {e}"
                fail_writer.writerow([vid, v.get("title"), reason])
                segments = []
                if not meta:
                    failed += 1
                    print(f"[{i}/{len(videos)}] FAIL {vid} - {reason[:80]}")
                    time.sleep(SLEEP_BETWEEN)
                    continue
                print(f"[{i}/{len(videos)}] meta-only {vid} - no transcript")

            text = "\n".join(s["text"] for s in segments)
            record = {
                "id": vid,
                "title": v.get("title"),
                "url": v.get("url"),
                **meta,
                "segments": segments,
                "text": text,
            }
            json_file.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (OUT_DIR / f"{vid}.txt").write_text(text, encoding="utf-8")
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            jsonl.flush()

            ok += 1
            print(f"[{i}/{len(videos)}] OK   {vid} - {len(segments)} segments")
            time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. success={ok} skipped(existing)={skipped} failed={failed}")
    print(f"Transcripts in ./{OUT_DIR}/  |  failures in {fail_path.name}")

    n = write_csv(OUT_DIR)
    print(f"Flattened {n} videos into {OUT_DIR / 'videos.csv'}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# SSL note
# ---------------------------------------------------------------------------
# If you see: CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate
# it is a local cert-bundle problem (common behind corporate proxies), not a
# YouTube block. Fixes, in order of preference:
#   1) pip install --upgrade certifi   (and rerun)
#   2) On macOS with python.org builds: run "Install Certificates.command"
#   3) Point Python at certifi's bundle before running:
#        import certifi, os
#        os.environ["SSL_CERT_FILE"] = certifi.where()
#      (add this at the top of main() if needed)
#   Avoid disabling verification outright unless you understand the risk.
