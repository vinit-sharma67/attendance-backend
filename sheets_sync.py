"""Google Sheets sync via Apps Script webhook (no Google Cloud needed).

How it works: the Google Sheet contains a small Apps Script web app.
After attendance is confirmed, the backend POSTs today's records to that
web app URL, and the script writes them into the sheet.

Render env vars:
  SHEETS_WEBHOOK_URL   = the Apps Script /exec URL
  SHEETS_WEBHOOK_TOKEN = same secret token as written inside the script

If SHEETS_WEBHOOK_URL is empty, sync is silently skipped.
Uses only Python stdlib (urllib) — no new dependencies.
"""
import json
import os
import urllib.request
from datetime import date

WEBHOOK_URL = os.getenv("SHEETS_WEBHOOK_URL", "")
WEBHOOK_TOKEN = os.getenv("SHEETS_WEBHOOK_TOKEN", "")


def sync_attendance(day: date, records: list[dict]) -> bool:
    """records: [{roll_no, name, status, confidence}] sorted by roll no.
    Returns True on success, False if disabled or failed.
    Never raises — attendance saving must not break because of Sheets."""
    if not WEBHOOK_URL:
        return False
    try:
        payload = json.dumps({
            "token": WEBHOOK_TOKEN,
            "date": day.isoformat(),          # e.g. 2026-07-22
            "records": [
                {"roll_no": r["roll_no"], "name": r["name"],
                 "status": r["status"]} for r in records
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            WEBHOOK_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", "ignore")
            print(f"[sheets_sync] webhook response: {resp.status} {body[:200]}")
            return resp.status == 200
    except Exception as e:
        print(f"[sheets_sync] failed: {e}")
        return False
