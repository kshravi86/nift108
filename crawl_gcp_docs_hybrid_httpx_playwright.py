# crawl_gcp_docs_hybrid_httpx_playwright.py
#
# Hybrid crawler:
# - Primary fetch via httpx (async, concurrent).
# - Fallback to Playwright (async Chromium render) only when the HTTP HTML
#   does not contain a devsite article container (div.devsite-article).
#
# Output: JSONL records with {area,url,title,content,fetch} (or {url,error,...}).

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

# Strict PMLE scope only. Keep this list aligned to the exam blueprint domains.
AREAS: dict[str, dict[str, str]] = {
    "vertex-ai": {
        "start_url": "https://docs.cloud.google.com/vertex-ai/docs",
        "path_prefix": "/vertex-ai/docs",
    },
    # Low-code ML: BigQuery ML lives under BigQuery docs.
    "bigquery": {
        # NOTE: The older BQML intro URL (`/bigquery/docs/bigqueryml-introduction`) now 404s.
        # Use a stable landing page that exposes the full nav so we can discover BQML pages.
        "start_url": "https://docs.cloud.google.com/bigquery/docs/introduction",
        "path_prefix": "/bigquery/docs",
    },
    "dataflow": {
        "start_url": "https://docs.cloud.google.com/dataflow/docs",
        "path_prefix": "/dataflow/docs",
    },
    "dataproc": {
        "start_url": "https://docs.cloud.google.com/dataproc/docs",
        "path_prefix": "/dataproc/docs",
    },
    # Cloud DLP docs redirect here (rebrand); keep the area name as "dlp" for exam mapping.
    "dlp": {
        "start_url": "https://docs.cloud.google.com/sensitive-data-protection/docs",
        "path_prefix": "/sensitive-data-protection/docs",
    },
    # Data platform choices used in PMLE solution design scenarios.
    "storage": {
        "start_url": "https://docs.cloud.google.com/storage/docs",
        "path_prefix": "/storage/docs",
    },
    "spanner": {
        "start_url": "https://docs.cloud.google.com/spanner/docs",
        "path_prefix": "/spanner/docs",
    },
    "cloudsql": {
        "start_url": "https://docs.cloud.google.com/sql/docs",
        "path_prefix": "/sql/docs",
    },
    "bigtable": {
        "start_url": "https://docs.cloud.google.com/bigtable/docs",
        "path_prefix": "/bigtable/docs",
    },
    "pubsub": {
        "start_url": "https://docs.cloud.google.com/pubsub/docs",
        "path_prefix": "/pubsub/docs",
    },
    "run": {
        "start_url": "https://docs.cloud.google.com/run/docs",
        "path_prefix": "/run/docs",
    },
    # Pipeline automation and MLOps operations.
    "build": {
        "start_url": "https://docs.cloud.google.com/build/docs",
        "path_prefix": "/build/docs",
    },
    "artifact-registry": {
        "start_url": "https://docs.cloud.google.com/artifact-registry/docs",
        "path_prefix": "/artifact-registry/docs",
    },
    "composer": {
        "start_url": "https://docs.cloud.google.com/composer/docs",
        "path_prefix": "/composer/docs",
    },
    "monitoring": {
        "start_url": "https://docs.cloud.google.com/monitoring/docs",
        "path_prefix": "/monitoring/docs",
    },
    "logging": {
        "start_url": "https://docs.cloud.google.com/logging/docs",
        "path_prefix": "/logging/docs",
    },
    "iam": {
        "start_url": "https://docs.cloud.google.com/iam/docs",
        "path_prefix": "/iam/docs",
    },
    "kms": {
        "start_url": "https://docs.cloud.google.com/kms/docs",
        "path_prefix": "/kms/docs",
    },
    "vpc": {
        "start_url": "https://docs.cloud.google.com/vpc/docs",
        "path_prefix": "/vpc/docs",
    },
    # ML APIs and managed AI products included in PMLE low-code architecture choices.
    "vision": {
        "start_url": "https://docs.cloud.google.com/vision/docs",
        "path_prefix": "/vision/docs",
    },
    "natural-language": {
        "start_url": "https://docs.cloud.google.com/natural-language/docs",
        "path_prefix": "/natural-language/docs",
    },
    "speech-to-text": {
        "start_url": "https://docs.cloud.google.com/speech-to-text/docs",
        "path_prefix": "/speech-to-text/docs",
    },
    "translate": {
        "start_url": "https://docs.cloud.google.com/translate/docs",
        "path_prefix": "/translate/docs",
    },
    "document-ai": {
        "start_url": "https://docs.cloud.google.com/document-ai/docs",
        "path_prefix": "/document-ai/docs",
    },
    "retail": {
        "start_url": "https://docs.cloud.google.com/retail/docs",
        "path_prefix": "/retail/docs",
    },
    # Accelerator choices for training.
    "tpu": {
        "start_url": "https://docs.cloud.google.com/tpu/docs",
        "path_prefix": "/tpu/docs",
    },
}

AREA_ORDER = [
    "vertex-ai",
    "bigquery",
    "dataflow",
    "dataproc",
    "dlp",
    "storage",
    "spanner",
    "cloudsql",
    "bigtable",
    "pubsub",
    "run",
    "build",
    "artifact-registry",
    "composer",
    "monitoring",
    "logging",
    "iam",
    "kms",
    "vpc",
    "vision",
    "natural-language",
    "speech-to-text",
    "translate",
    "document-ai",
    "retail",
    "tpu",
]

ALLOWED_DOMAIN = "docs.cloud.google.com"

DEFAULT_MAX_PAGES = 2000
DEFAULT_HTTP_CONCURRENCY = 10
DEFAULT_PLAYWRIGHT_CONCURRENCY = 2
DEFAULT_SLEEP_SECS = 0.2
DEFAULT_SAVE_EVERY = 10

DEFAULT_OUTPUT_PATH = "gcp_docs_pmle_hybrid.jsonl"
DEFAULT_STATE_PATH = "gcp_crawl_state_pmle_hybrid.json"
DEFAULT_LOG_PATH = "crawl_gcp_docs_hybrid_pmle.log"

UA = "Mozilla/5.0 (compatible; DocCrawler/2.0; +https://github.com/kshravi86/nift108)"


def log(log_path: str, *args: Any) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = " ".join(str(a) for a in args)
    line = f"{ts} {msg}"
    try:
        print(line, flush=True)
    except OSError:
        pass
    try:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
    except Exception:
        pass


def canonicalize(url: str) -> str:
    parts = urlsplit(url)
    scheme = "https"
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunsplit((scheme, netloc, path, "", ""))


def area_for_url(url: str) -> str | None:
    parts = urlsplit(url)
    if parts.netloc.lower() != ALLOWED_DOMAIN:
        return None
    for area, cfg in AREAS.items():
        if parts.path.startswith(cfg["path_prefix"]):
            return area
    return None


def is_allowed(url: str) -> bool:
    return area_for_url(url) is not None


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("div.devsite-article") or soup.select_one("main") or soup.body
    if not main:
        return ""
    for tag in main.select("nav, aside, footer, script, style"):
        tag.decompose()
    text = main.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def extract_links(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        abs_url = canonicalize(urljoin(base_url, href))
        if is_allowed(abs_url):
            links.add(abs_url)
    return links


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    t = soup.title.string if soup.title else ""
    return (t or "").strip()


def has_devsite_article(html: str) -> bool:
    # Cheap heuristic: devsite pages include this container when the real article is present.
    return "devsite-article" in html


def load_state(state_path: str) -> tuple[set[str], dict[str, deque[str]]] | None:
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        visited = set(data.get("visited", []))
        queues: dict[str, deque[str]] = {area: deque() for area in AREAS.keys()}
        if "queues" in data and isinstance(data["queues"], dict):
            for area in AREAS.keys():
                queues[area] = deque(data["queues"].get(area, []))
        else:
            q = deque(data.get("queue", []))
            for u in q:
                a = area_for_url(u)
                if a:
                    queues[a].append(u)
        return visited, queues
    except Exception:
        return None


def save_state(state_path: str, visited: set[str], queues: dict[str, deque[str]]) -> None:
    data = {
        "visited": list(visited),
        "queues": {area: list(q) for area, q in queues.items()},
        "saved_at": time.time(),
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def seed_visited_from_output(output_path: str, visited: set[str]) -> None:
    if not os.path.exists(output_path):
        return
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    url = obj.get("url")
                    if url:
                        visited.add(canonicalize(url))
                except Exception:
                    continue
    except Exception:
        pass


@dataclass(frozen=True)
class FetchResult:
    url: str
    html: str
    title: str
    fetch: str  # "httpx" | "playwright"


async def fetch_httpx(client: httpx.AsyncClient, url: str) -> tuple[str, str]:
    # Returns (final_url, html)
    backoff = 1.0
    for attempt in range(1, 4):
        try:
            r = await client.get(url, follow_redirects=True)
            if r.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"HTTP {r.status_code}", request=r.request, response=r
                )
            return str(r.url), r.text
        except Exception:
            if attempt >= 3:
                raise
            await asyncio.sleep(backoff + random.random() * 0.25)
            backoff *= 2
    raise RuntimeError("unreachable")


async def fetch_playwright(context, url: str) -> tuple[str, str, str]:
    # Returns (final_url, title, html)
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        try:
            await page.wait_for_selector("div.devsite-article", timeout=15000)
        except PlaywrightTimeoutError:
            await page.wait_for_selector("body", timeout=15000)
        final_url = canonicalize(page.url)
        title = (await page.title()) or ""
        html = await page.content()
        return final_url, title, html
    finally:
        await page.close()


async def fetch_with_fallback(
    *,
    client: httpx.AsyncClient,
    pw_context,
    pw_sem: asyncio.Semaphore,
    url: str,
) -> FetchResult:
    final_url, html = await fetch_httpx(client, url)
    final_url = canonicalize(final_url)
    if has_devsite_article(html):
        title = extract_title(html)
        return FetchResult(url=final_url, html=html, title=title, fetch="httpx")

    # Fallback to Playwright only when the HTTP response doesn't contain the devsite article.
    async with pw_sem:
        f_url, title, html2 = await fetch_playwright(pw_context, final_url)
    return FetchResult(url=f_url, html=html2, title=title, fetch="playwright")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid GCP Docs crawler (httpx + Playwright fallback).")
    p.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    p.add_argument("--http-concurrency", type=int, default=DEFAULT_HTTP_CONCURRENCY)
    p.add_argument("--playwright-concurrency", type=int, default=DEFAULT_PLAYWRIGHT_CONCURRENCY)
    p.add_argument("--sleep-secs", type=float, default=DEFAULT_SLEEP_SECS)
    p.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    p.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    p.add_argument("--state-path", default=DEFAULT_STATE_PATH)
    p.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    return p.parse_args()


async def crawl_async(args: argparse.Namespace) -> None:
    state = load_state(args.state_path)
    if state:
        visited, queues = state
        seed_visited_from_output(args.output_path, visited)
        for area in AREAS.keys():
            if queues.get(area):
                queues[area] = deque(u for u in queues[area] if u not in visited)
        resumed = True
    else:
        visited = set()
        queues = {area: deque([canonicalize(cfg["start_url"])]) for area, cfg in AREAS.items()}
        seed_visited_from_output(args.output_path, visited)
        resumed = False

    queued_set: set[str] = set()
    for q in queues.values():
        queued_set.update(q)

    log(args.log_path, f"Resume: {resumed}")
    log(args.log_path, "Start queue length:", sum(len(q) for q in queues.values()))
    log(args.log_path, f"Visited (seeded): {len(visited)}")
    for area in AREA_ORDER:
        log(args.log_path, f"  queue[{area}]: {len(queues[area])}")

    # If we have visited but nothing queued, bootstrap from start URLs.
    needs_bootstrap = not any(u not in visited for u in queued_set)

    data_lock = asyncio.Lock()
    save_lock = asyncio.Lock()
    rr_idx = 0
    processed = 0
    pages_since_save = 0

    file_mode = "a" if resumed or os.path.exists(args.output_path) else "w"

    async def writer_loop(q: asyncio.Queue[dict[str, Any]]) -> None:
        with open(args.output_path, file_mode, encoding="utf-8") as f:
            while True:
                item = await q.get()
                if item is None:  # type: ignore[comparison-overlap]
                    q.task_done()
                    break
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                f.flush()
                q.task_done()

    record_q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=1000)
    writer_task = asyncio.create_task(writer_loop(record_q))

    async def maybe_save_state() -> None:
        nonlocal pages_since_save
        if pages_since_save < args.save_every:
            return
        async with save_lock:
            if pages_since_save < args.save_every:
                return
            snap_visited: set[str]
            snap_queues: dict[str, deque[str]]
            async with data_lock:
                snap_visited = set(visited)
                snap_queues = {a: deque(list(q)) for a, q in queues.items()}
            await asyncio.to_thread(save_state, args.state_path, snap_visited, snap_queues)
            pages_since_save = 0
            log(args.log_path, "State saved. Visited:", len(snap_visited))

    async def next_url() -> tuple[str, str] | None:
        nonlocal rr_idx
        async with data_lock:
            if len(visited) >= args.max_pages:
                return None
            picked_area = None
            for _ in range(len(AREA_ORDER)):
                area = AREA_ORDER[rr_idx % len(AREA_ORDER)]
                rr_idx += 1
                if queues[area]:
                    picked_area = area
                    break
            if not picked_area:
                return None
            url = queues[picked_area].popleft()
            queued_set.discard(url)
            if url in visited:
                return None
            visited.add(url)
            return picked_area, url

    limits = httpx.Limits(
        max_connections=max(args.http_concurrency * 2, 20),
        max_keepalive_connections=max(args.http_concurrency, 10),
    )
    timeout = httpx.Timeout(30.0, connect=10.0)

    pw_sem = asyncio.Semaphore(args.playwright_concurrency)

    async with httpx.AsyncClient(headers={"User-Agent": UA}, limits=limits, timeout=timeout) as client:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=UA,
                viewport={"width": 1400, "height": 900},
            )

            # Bootstrap queue links if needed.
            if needs_bootstrap:
                log(args.log_path, "Bootstrapping queues from start URLs (no unvisited URLs queued)...")
                for area in AREA_ORDER:
                    start_url = canonicalize(AREAS[area]["start_url"])
                    try:
                        fr = await fetch_with_fallback(
                            client=client, pw_context=context, pw_sem=pw_sem, url=start_url
                        )
                        links = extract_links(fr.html, fr.url)
                        added = 0
                        async with data_lock:
                            for link in links:
                                a2 = area_for_url(link)
                                if not a2:
                                    continue
                                if link in visited or link in queued_set:
                                    continue
                                queues[a2].append(link)
                                queued_set.add(link)
                                added += 1
                        log(args.log_path, f"  bootstrap[{area}] added_links={added} fetch={fr.fetch}")
                        await asyncio.sleep(args.sleep_secs)
                    except Exception as e:
                        log(args.log_path, f"Bootstrap error for {start_url}: {e}")

            async def worker(worker_id: int) -> None:
                nonlocal processed, pages_since_save
                while True:
                    nxt = await next_url()
                    if not nxt:
                        return
                    area, url = nxt
                    async with data_lock:
                        processed += 1
                        cur = processed
                    log(args.log_path, f"[{cur}/{args.max_pages}] ({area}) Fetch: {url}")

                    try:
                        fr = await fetch_with_fallback(
                            client=client, pw_context=context, pw_sem=pw_sem, url=url
                        )
                        content = extract_main_text(fr.html)
                        record = {
                            "area": area,
                            "url": fr.url,
                            "title": fr.title,
                            "content": content,
                            "fetch": fr.fetch,
                        }
                        await record_q.put(record)

                        links = extract_links(fr.html, fr.url)
                        async with data_lock:
                            for link in links:
                                a2 = area_for_url(link)
                                if not a2:
                                    continue
                                if link in visited or link in queued_set:
                                    continue
                                queues[a2].append(link)
                                queued_set.add(link)
                            q_sizes = {a: len(queues[a]) for a in AREA_ORDER}
                            v_count = len(visited)
                        log(args.log_path, "Queued:", q_sizes, "| Visited:", v_count, "| Title:", fr.title[:80])
                    except Exception as e:
                        err = {"url": url, "area": area, "error": str(e), "fetch": "httpx_or_playwright"}
                        await record_q.put(err)
                        log(args.log_path, f"Error: {e}")
                    finally:
                        pages_since_save += 1
                        await maybe_save_state()
                        await asyncio.sleep(args.sleep_secs)

            workers = [asyncio.create_task(worker(i)) for i in range(args.http_concurrency)]
            await asyncio.gather(*workers)

            await browser.close()

    await asyncio.to_thread(save_state, args.state_path, visited, queues)
    log(args.log_path, "Done. Visited:", len(visited), "| Remaining queues:", {a: len(queues[a]) for a in AREA_ORDER})

    await record_q.put(None)
    await record_q.join()
    await writer_task


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(crawl_async(args))
    except KeyboardInterrupt:
        raise
    except Exception:
        import traceback

        log(args.log_path, "FATAL:", traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

