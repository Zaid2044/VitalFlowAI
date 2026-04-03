"""
One-time migration: encrypt existing plaintext vitals in the readings table.

Run ONCE after setting FIELD_ENCRYPTION_KEY in your .env file.
The script operates on the live database file — back it up first.

Usage:
    cd backend
    python scripts/migrate_encrypt.py            # dry-run (no changes)
    python scripts/migrate_encrypt.py --apply    # write encrypted values

How it works:
    1. Read every row in the `readings` table using raw SQLite (no ORM),
       so the plaintext float values come back unmodified.
    2. For each vital column, encrypt the value with Fernet and UPDATE the row.
    3. Print a summary of rows processed.

After the migration is complete, update models.py to use EncryptedFloat:
    from crypto import EncryptedFloat
    blood_sugar = Column(EncryptedFloat, nullable=True)
    # ... repeat for systolic_bp, diastolic_bp, heart_rate, temperature, spo2

WARNING: Do NOT restart the application between running this script and
updating models.py — or reads will fail because the DB has ciphertext but
the ORM still expects plain floats.
"""

import os
import sys
import sqlite3

# Add backend/ to path so we can import crypto
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from crypto import encrypt_value  # noqa: E402 — after sys.path setup

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vitalflow.db")

VITAL_COLUMNS = [
    "blood_sugar",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "temperature",
    "spo2",
]


def already_encrypted(value: str) -> bool:
    """
    Fernet ciphertext starts with "gAAAAA" (the base64-encoded version token).
    If a value already looks like ciphertext, skip it to allow re-running safely.
    """
    return isinstance(value, str) and value.startswith("gAAAAA")


def migrate(apply: bool = False):
    if not os.path.exists(DB_PATH):
        print(f"[MIGRATE] Database not found at {DB_PATH}")
        sys.exit(1)

    print(f"[MIGRATE] Database : {DB_PATH}")
    print(f"[MIGRATE] Mode     : {'APPLY (writing changes)' if apply else 'DRY RUN (no changes)'}")
    print(f"[MIGRATE] Columns  : {', '.join(VITAL_COLUMNS)}")
    print()

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id, " + ", ".join(VITAL_COLUMNS) + " FROM readings ORDER BY id")
    rows = cursor.fetchall()

    processed = 0
    skipped   = 0
    errors    = 0

    for row in rows:
        row_id = row[0]
        updates: dict[str, str] = {}

        for i, col in enumerate(VITAL_COLUMNS):
            raw = row[i + 1]
            if raw is None:
                continue
            if already_encrypted(str(raw)):
                skipped += 1
                continue
            try:
                encrypted = encrypt_value(float(raw))
                updates[col] = encrypted
            except Exception as exc:
                print(f"  [ERROR] Row {row_id}, column {col}: {exc}")
                errors += 1

        if not updates:
            continue

        set_clause = ", ".join(f"{col} = ?" for col in updates)
        values     = list(updates.values()) + [row_id]

        if apply:
            cursor.execute(
                f"UPDATE readings SET {set_clause} WHERE id = ?", values
            )
        else:
            print(
                f"  [DRY RUN] Row {row_id}: would encrypt {list(updates.keys())}"
            )

        processed += 1

    if apply:
        conn.commit()
        print(f"\n[MIGRATE] Done. {processed} rows encrypted, {skipped} already encrypted, {errors} errors.")
    else:
        print(f"\n[MIGRATE] Dry run complete. {processed} rows would be encrypted.")
        print("[MIGRATE] Re-run with --apply to commit changes.")

    conn.close()

    # Change the column types in the schema so SQLAlchemy accepts Text storage
    if apply:
        print("\n[MIGRATE] Next step: update models.py — replace Column(Float) with Column(EncryptedFloat)")
        print("          for: blood_sugar, systolic_bp, diastolic_bp, heart_rate, temperature, spo2")


if __name__ == "__main__":
    apply = "--apply" in sys.argv
    migrate(apply=apply)
