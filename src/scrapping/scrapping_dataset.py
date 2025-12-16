from google_play_scraper import reviews, Sort
import pandas as pd
import time, datetime as dt
from pathlib import Path

# === [1] Path Setup ===
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MASTER_PATH = DATA_DIR / "master" / "floq_reviews_master.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)

# === [2] Config ===
APP_ID = "id.kriptomaksima.app"
LANG = "id"
COUNTRY = "id"
BATCH = 200
MAX_PAGES = 200

# === [3] Load master lama ===
if MASTER_PATH.exists():
    master = pd.read_csv(MASTER_PATH)
    seen = set(master.get("reviewId", []).astype(str))
else:
    master = pd.DataFrame()
    seen = set()

# === [4] Scraping incremental ===
all_new = []
continuation_token = None
pages = 0

print(f"[{dt.datetime.now()}] Start scraping {APP_ID}")

while True:
    try:
        batch, continuation_token = reviews(
            APP_ID,
            lang=LANG,
            country=COUNTRY,
            sort=Sort.NEWEST,
            count=BATCH,
            continuation_token=continuation_token
        )
    except Exception as e:
        print("⚠️ Fetch error:", e)
        break   # ⬅️ JANGAN retry infinite di CI

    pages += 1
    if not batch:
        break

    fresh = [r for r in batch if str(r.get("reviewId")) not in seen]
    all_new.extend(fresh)

    print(f"Page {pages}: new {len(fresh)} | total_new {len(all_new)}")

    if len(fresh) == 0 or pages >= MAX_PAGES or continuation_token is None:
        break

    time.sleep(0.5)

# === [5] Save RAW snapshot ===
today = dt.datetime.now().strftime("%Y%m%d")
raw_path = RAW_DIR / f"floq_reviews_{today}.csv"

if all_new:
    df_new = pd.DataFrame(all_new)
    df_new.to_csv(raw_path, index=False)
    print(f"✅ RAW saved: {raw_path}")
else:
    df_new = pd.DataFrame()
    print("ℹ️ No new reviews today.")

# === [6] Merge ke master ===
if not df_new.empty:
    master = pd.concat([master, df_new], ignore_index=True)
    if "reviewId" in master.columns:
        master.drop_duplicates(subset=["reviewId"], inplace=True)
    master.to_csv(MASTER_PATH, index=False)
    print(f"✅ MASTER updated ({len(master)} rows)")
else:
    if not MASTER_PATH.exists():
        master.to_csv(MASTER_PATH, index=False)
        print("⚠️ MASTER initialized (empty)")
    else:
        print("MASTER unchanged.")

print("🎉 Scraping step finished successfully.")
