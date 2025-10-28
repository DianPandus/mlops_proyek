from google_play_scraper import reviews, Sort
import pandas as pd
import os, time, datetime as dt
from pathlib import Path

# === [1] Path Setup (biar gak nyasar ke folder src) ===
ROOT_DIR = Path(__file__).resolve().parents[2]  # naik 2 folder ke root proyek
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MASTER_PATH = DATA_DIR / "master" / "floq_reviews_master.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)

# === [2] Config scraping ===
APP_ID = "id.kriptomaksima.app"   # Floq
LANG = "id"
COUNTRY = "id"
BATCH = 200
MAX_PAGES = 200

# === [3] Load master lama kalau ada ===
if MASTER_PATH.exists():
    master = pd.read_csv(MASTER_PATH)
    seen = set(master.get("reviewId", []).astype(str))
else:
    master = pd.DataFrame()
    seen = set()

# === [4] Loop incremental ===
all_new = []
continuation_token = None
pages = 0

print(f"[{dt.datetime.now()}] Start incremental fetch for {APP_ID}")
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
        print("Fetch error, retrying in 3s:", e)
        time.sleep(3)
        continue

    pages += 1
    if not batch:
        print("Empty batch, stop.")
        break

    fresh = [r for r in batch if str(r.get("reviewId")) not in seen]
    all_new.extend(fresh)

    print(f"Page {pages}: got {len(batch)} | new {len(fresh)} | total_new {len(all_new)}")

    if len(fresh) == 0:
        print("No new reviews -> early stop.")
        break

    if continuation_token is None or pages >= MAX_PAGES:
        print("No more pages or hit MAX_PAGES.")
        break

    time.sleep(0.5)

# === [5] Simpan snapshot raw harian ===
today = dt.datetime.now().strftime("%Y%m%d")
raw_path = RAW_DIR / f"floq_reviews_{today}.csv"

if all_new:
    df_new = pd.DataFrame(all_new)
    df_new.to_csv(raw_path, index=False)
    print(f"✅ Saved RAW snapshot: {raw_path} ({len(df_new)} rows)")
else:
    df_new = pd.DataFrame()
    print("No NEW data today. RAW snapshot not created.")

# === [6] Merge ke master + dedup ===
if len(df_new) > 0:
    master = pd.concat([master, df_new], ignore_index=True)
    if "reviewId" in master.columns:
        before = len(master)
        master.drop_duplicates(subset=["reviewId"], keep="first", inplace=True)
        after = len(master)
        print(f"Dedup master: {before} -> {after}")
    master.to_csv(MASTER_PATH, index=False)
    print(f"✅ MASTER updated: {MASTER_PATH} (rows={len(master)})")
else:
    print("MASTER unchanged (no new rows).")

# === [7] Preview hasil ===
if len(master) > 0:
    cols = ["reviewId", "score", "at", "userName", "content"]
    print("\n--- Preview Data (Last 10) ---")
    print(master[[c for c in cols if c in master.columns]].tail(10))
