# crawl_gcp_docs_playwright.py
import json
import time
import re
import os
from collections import deque
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# Strict PMLE scope only. Keep this list aligned to the exam blueprint domains.
AREAS = {
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
ALLOWED_PATH_PREFIXES = [cfg["path_prefix"] for cfg in AREAS.values()]

MAX_PAGES = 10000  # safety cap
SLEEP_SECS = 1.0  # politeness delay
OUTPUT_PATH = "gcp_docs_pmle_strict.jsonl"
STATE_PATH = "gcp_crawl_state_pmle_strict.json"
SAVE_EVERY = 5    # save resume state every N pages

# Best-effort logging. On Windows, background processes can sometimes lose a valid stdout handle.
# We write to a log file as well so progress can still be monitored.
LOG_PATH = "crawl_gcp_docs_playwright_pmle_strict.log"


def log(*args):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = " ".join(str(a) for a in args)
    line = f"{ts} {msg}"
    try:
        print(line, flush=True)
    except OSError:
        # stdout/stderr can be invalid in some detached process setups on Windows.
        pass
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(line + "\n")
    except Exception:
        # Logging must never break the crawl.
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


def extract_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        abs_url = urljoin(base_url, href)
        abs_url = canonicalize(abs_url)
        if is_allowed(abs_url):
            links.add(abs_url)
    return links


def load_state():
    if not os.path.exists(STATE_PATH):
        return None
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        visited = set(data.get("visited", []))
        if "queues" in data and isinstance(data["queues"], dict):
            queues = {area: deque(data["queues"].get(area, [])) for area in AREAS.keys()}
            return visited, queues
        # Back-compat: old single queue format.
        q = deque(data.get("queue", []))
        queues = {area: deque() for area in AREAS.keys()}
        for u in q:
            a = area_for_url(u)
            if a:
                queues[a].append(u)
        return visited, queues
    except Exception:
        return None


def save_state(visited, queues):
    data = {
        "visited": list(visited),
        "queues": {area: list(q) for area, q in queues.items()},
        "saved_at": time.time(),
    }
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)


def seed_visited_from_output(visited):
    if not os.path.exists(OUTPUT_PATH):
        return
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
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


def crawl():
    state = load_state()
    if state:
        visited, queues = state
        # Reconcile state with what's already written to disk. If the last run crashed before
        # persisting STATE_PATH, OUTPUT_PATH may contain additional URLs that would otherwise
        # be re-crawled as duplicates.
        seed_visited_from_output(visited)
        for area in AREAS.keys():
            if queues.get(area):
                queues[area] = deque(u for u in queues[area] if u not in visited)
        resumed = True
    else:
        visited = set()
        queues = {area: deque([canonicalize(cfg["start_url"])]) for area, cfg in AREAS.items()}
        seed_visited_from_output(visited)
        resumed = False

    log(f"Resume: {resumed}")
    log("Start queue length:", sum(len(q) for q in queues.values()))
    log(f"Visited (seeded): {len(visited)}")
    for area in AREA_ORDER:
        log(f"  queue[{area}]: {len(queues[area])}")

    file_mode = "a" if resumed or os.path.exists(OUTPUT_PATH) else "w"
    pages_since_save = 0
    processed = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (compatible; DocCrawler/1.0)",
            viewport={"width": 1400, "height": 900},
        )
        page = context.new_page()

        # If state is missing but OUTPUT_PATH exists, we only have a visited set. In that case,
        # re-fetch the start URLs once to rebuild a queue of unvisited links without emitting duplicates.
        queued_set = set()
        for q in queues.values():
            queued_set.update(q)

        needs_bootstrap = not any(u not in visited for u in queued_set)
        if needs_bootstrap:
            log("Bootstrapping queues from start URLs (no unvisited URLs queued)...")
            for area in AREA_ORDER:
                start_url = canonicalize(AREAS[area]["start_url"])
                try:
                    log(f"  bootstrap[{area}]: {start_url}")
                    page.goto(start_url, wait_until="domcontentloaded", timeout=30000)
                    try:
                        page.wait_for_selector("div.devsite-article", timeout=15000)
                    except PlaywrightTimeoutError:
                        page.wait_for_selector("body", timeout=15000)

                    html = page.content()
                    added = 0
                    for link in extract_links(html, start_url):
                        a2 = area_for_url(link)
                        if not a2:
                            continue
                        if link in visited or link in queued_set:
                            continue
                        queues[a2].append(link)
                        queued_set.add(link)
                        added += 1
                    log(f"    added_links: {added}")
                    time.sleep(SLEEP_SECS)
                except Exception as e:
                    log(f"Bootstrap error for {start_url}: {e}")
            log("Bootstrap queues:", {a: len(queues[a]) for a in AREA_ORDER})

        with open(OUTPUT_PATH, file_mode, encoding="utf-8") as f:
            rr_idx = 0
            while len(visited) < MAX_PAGES and any(queues[a] for a in AREA_ORDER):
                # Round-robin across doc areas to avoid one area (Vertex AI) starving the others.
                picked_area = None
                for _ in range(len(AREA_ORDER)):
                    area = AREA_ORDER[rr_idx % len(AREA_ORDER)]
                    rr_idx += 1
                    if queues[area]:
                        picked_area = area
                        break
                if not picked_area:
                    break

                url = queues[picked_area].popleft()
                queued_set.discard(url)
                if url in visited:
                    continue
                visited.add(url)
                processed += 1
                log(f"[{processed}/{MAX_PAGES}] ({picked_area}) Fetch: {url}")

                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    try:
                        page.wait_for_selector("div.devsite-article", timeout=15000)
                    except PlaywrightTimeoutError:
                        page.wait_for_selector("body", timeout=15000)

                    html = page.content()
                    title = page.title() or ""
                    content = extract_main_text(html)

                    record = {"area": picked_area, "url": url, "title": title, "content": content}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

                    for link in extract_links(html, url):
                        a2 = area_for_url(link)
                        if not a2:
                            continue
                        if link in visited or link in queued_set:
                            continue
                        queues[a2].append(link)
                        queued_set.add(link)
                    log(
                        "Queued:",
                        {a: len(queues[a]) for a in AREA_ORDER},
                        "| Visited:",
                        len(visited),
                        "| Title:",
                        title[:80],
                    )

                    time.sleep(SLEEP_SECS)

                except Exception as e:
                    err = {"url": url, "error": str(e)}
                    f.write(json.dumps(err, ensure_ascii=False) + "\n")
                    f.flush()
                    log(f"Error: {e}")
                finally:
                    pages_since_save += 1
                    if pages_since_save >= SAVE_EVERY:
                        save_state(visited, queues)
                        log(
                            "State saved. Queues:",
                            {a: len(queues[a]) for a in AREA_ORDER},
                            "| Visited:",
                            len(visited),
                        )
                        pages_since_save = 0

        browser.close()

    save_state(visited, queues)
    log(
        "Done. Visited:",
        len(visited),
        "| Remaining queues:",
        {a: len(queues[a]) for a in AREA_ORDER},
    )


if __name__ == "__main__":
    try:
        crawl()
    except Exception:
        import traceback

        log("FATAL:", traceback.format_exc())
        raise
