import os, sys, json, re
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

CONTEXT_URLS = {
    "monolog": "https://www.uclass.psychol.ucl.ac.uk/Release2/Monologue/AudioOnly/wav/",
    "reading": "https://www.uclass.psychol.ucl.ac.uk/Release2/Reading/AudioOnly/wav/",
    "conversation": "https://www.uclass.psychol.ucl.ac.uk/Release2/Conversation/AudioOnly/wav/",
}

DATASET_ID = "uclass2"
SRC = "en"
TGTS = ["es","it","fr","de","nl","pt"]
HEADERS = {"User-Agent": "Mozilla/5.0"}

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def download_file(url, dest, referer=None):
    headers = dict(HEADERS)
    if referer: headers["Referer"] = referer
    with requests.get(url, stream=True, headers=headers, timeout=(15,90)) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(262144):
                if chunk:
                    f.write(chunk)

def get_wav_links(page_url):
    """Return deduplicated list of absolute wav URLs found on the page (hrefs ending with .wav)."""
    resp = requests.get(page_url, headers=HEADERS, timeout=(15,90))
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(page_url, a["href"].strip())
        if urlparse(href).path.lower().endswith(".wav"):
            if href not in links:
                links.append(href)
    return links

def main():
    base_dir = os.environ.get("H2T_DATADIR")
    if not base_dir:
        print("ERROR: set H2T_DATADIR environment variable before running (PowerShell: $env:H2T_DATADIR = 'C:\\path')", file=sys.stderr)
        sys.exit(1)
    # gather all links first, in ordered by context
    ordered = []   # list of (ctx, url)
    seen = set()
    for ctx in ("monolog","reading","conversation"):
        page = CONTEXT_URLS[ctx]
        try:
            links = get_wav_links(page)
        except Exception as e:
            print(f"WARNING: could not fetch links for {ctx} ({page}): {e}", file=sys.stderr)
            links = []
        for url in links:
            if url not in seen:
                seen.add(url)
                ordered.append((ctx, url))

    total = len(ordered)
    if total == 0:
        print("No .wav links found across contexts; exiting.", file=sys.stderr)
        return

    id_width = max(1, len(str(total)))
    print(f"Total unique .wav files found across contexts: {total} (id width {id_width})")

    all_records = []
    for i, (ctx, url) in enumerate(ordered, start=1):
        sid = str(i).zfill(id_width)
        fname = Path(urlparse(url).path).name.split("?")[0] or (sid + ".wav")
        stage_dir = Path(base_dir) / DATASET_ID / "audio" / SRC / ctx
        ensure_dir(stage_dir)
        dst = stage_dir / fname

        if not dst.exists():
            try:
                download_file(url, dst, referer=CONTEXT_URLS[ctx])
                print("Downloaded:", ctx, fname)
            except Exception as e:
               print("Download failed:", url, e, file=sys.stderr)
               continue
            
           
        else:
            print("Exists, skipped:", fname)

        pid = None
        m = re.search(r"(?:spk|speaker|child|kid|p|s)[-_]*?(\d{1,3})", fname, re.I)
        if m:
            pid = m.group(1)

        rec = {
            "dataset_id": DATASET_ID,
            "sample_id": sid,
            "src_audio": f"/{DATASET_ID}/audio/{SRC}/{ctx}/{fname}",
            "src_ref": None,
            "tgt_ref": None,
            "src_lang": SRC,
            "benchmark_metadata": {
                "native_acc": None,
                "spoken_acc": None,
                "participant_id": pid,
                "context": ctx
            }
        }
        all_records.append(rec)


    # write all files for each target language
    manifests_base = Path("manifests") / DATASET_ID
    ensure_dir(manifests_base)
    for tgt in TGTS:
        outp = manifests_base / f"{SRC}-{tgt}.jsonl"
        with outp.open("w", encoding="utf-8") as fo:
            for r in all_records:
                ordered_rec = {
                    "dataset_id": r["dataset_id"],
                    "sample_id": r["sample_id"],
                    "src_audio": r["src_audio"],
                    "src_ref": r["src_ref"],
                    "tgt_ref": r["tgt_ref"],
                    "src_lang": r["src_lang"],
                    "tgt_lang": tgt,
                    "benchmark_metadata": r["benchmark_metadata"],
                }
                fo.write(json.dumps(ordered_rec, ensure_ascii=False) + "\n")

        print(outp, " is ready with  ", len(all_records), " records.")
    

if __name__ == "__main__":
    main()
