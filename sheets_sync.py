"""Optional Google Sheets sync.

After attendance is confirmed, today's record is also written to a Google
Sheet so teachers can view it without opening the app.

Setup (one time, ~10 min):
  1. Go to https://console.cloud.google.com -> create a project
  2. Enable "Google Sheets API"
  3. Create a Service Account -> create a JSON key -> download it
  4. Save the file as backend/service_account.json
  5. Create a Google Sheet, share it (Editor) with the service account
     email (looks like xxx@yyy.iam.gserviceaccount.com)
  6. Put the Sheet ID (from its URL) in .env as GOOGLE_SHEET_ID

If GOOGLE_SHEET_ID is not set, sync is silently skipped — the app works
fine without it.
"""
import os
from datetime import date

import gspread

SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "")
KEY_FILE = os.getenv("GOOGLE_KEY_FILE", "service_account.json")

_client = None


def _get_sheet():
    global _client
    if _client is None:
        _client = gspread.service_account(filename=KEY_FILE)
    return _client.open_by_key(SHEET_ID)


def sync_attendance(day: date, records: list[dict]) -> bool:
    """records: [{roll_no, name, status, confidence}] sorted by roll no.

    Layout: one worksheet per month (e.g. "Jul 2026").
    Column A = roll, B = name, then one column per day with P / A.
    Returns True on success, False if sync is disabled or failed.
    """
    if not SHEET_ID:
        return False
    try:
        book = _get_sheet()
        tab_name = day.strftime("%b %Y")
        try:
            ws = book.worksheet(tab_name)
        except gspread.WorksheetNotFound:
            ws = book.add_worksheet(tab_name, rows=200, cols=40)
            ws.update("A1:B1", [["Roll No", "Name"]])
            ws.format("1:1", {"textFormat": {"bold": True}})

        # Ensure all students are listed in column A/B
        existing = ws.col_values(1)[1:]  # skip header
        roll_to_row = {r: i + 2 for i, r in enumerate(existing)}
        new_rows = [[r["roll_no"], r["name"]] for r in records
                    if r["roll_no"] not in roll_to_row]
        if new_rows:
            ws.append_rows(new_rows)
            existing = ws.col_values(1)[1:]
            roll_to_row = {r: i + 2 for i, r in enumerate(existing)}

        # Find or create today's column
        header = ws.row_values(1)
        day_label = day.strftime("%d")
        if day_label in header:
            col = header.index(day_label) + 1
        else:
            col = len(header) + 1
            ws.update_cell(1, col, day_label)

        # Write P / A for every student in one batch
        cells = []
        for r in records:
            row = roll_to_row.get(r["roll_no"])
            if row:
                cells.append(gspread.Cell(row, col,
                             "P" if r["status"] == "present" else "A"))
        if cells:
            ws.update_cells(cells)
        return True
    except Exception as e:  # never break attendance saving because of Sheets
        print(f"[sheets_sync] failed: {e}")
        return False
