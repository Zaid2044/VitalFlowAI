"""
Field-Level Encryption for VitalFlow AI.

Problem
-------
SQLite stores data as plaintext on disk.  If the database file (vitalflow.db)
is stolen, every vital sign reading is immediately readable.  Even though the
database is not publicly accessible, defence-in-depth best practice for health
data is to encrypt sensitive columns so that physical access to the file does
not equal data access.

Solution
--------
Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256, authenticated) from
the `cryptography` library.  Each float value is:

    1. Converted to a UTF-8 string  → b"142.5"
    2. Encrypted with Fernet        → URL-safe base64 ciphertext
    3. Stored in the database as a  Text column

On read, the reverse happens transparently through a SQLAlchemy TypeDecorator.
The encryption key is a 32-byte URL-safe base64 string stored in the .env file
as FIELD_ENCRYPTION_KEY (never in the codebase).

Generating a key (run once, add to .env):
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Usage in models.py
------------------
Replace:
    blood_sugar = Column(Float, nullable=True)

With:
    blood_sugar = Column(EncryptedFloat, nullable=True)

Column storage type changes from Float → Text, so existing databases require
the migration script at backend/scripts/migrate_encrypt.py before switching.

Performance
-----------
Fernet encryption adds ~0.05 ms per field per read/write on a modern CPU —
negligible for a monitoring system with reads occurring every few minutes.
"""

import os
import logging
from typing import Optional

from sqlalchemy import String
from sqlalchemy.types import TypeDecorator

logger = logging.getLogger(__name__)

# ── Key Management ─────────────────────────────────────────────────────────────

def _load_fernet():
    """
    Load and cache a Fernet instance from the FIELD_ENCRYPTION_KEY env var.
    Raises RuntimeError if the key is missing or malformed (fail-fast).
    """
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise ImportError(
            "cryptography package not installed. "
            "Run: pip install cryptography"
        ) from exc

    key = os.getenv("FIELD_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError(
            "FIELD_ENCRYPTION_KEY environment variable is not set.\n"
            "Generate one with: python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\"\n"
            "Then add it to your .env file."
        )
    try:
        return Fernet(key.encode())
    except Exception as exc:
        raise RuntimeError(
            f"FIELD_ENCRYPTION_KEY is not a valid Fernet key: {exc}"
        ) from exc


# Module-level Fernet instance (initialised lazily on first use)
_fernet = None


def _get_fernet():
    global _fernet
    if _fernet is None:
        _fernet = _load_fernet()
    return _fernet


# ── Core encrypt / decrypt ─────────────────────────────────────────────────────

def encrypt_value(value: float) -> str:
    """
    Encrypt a float to a URL-safe base64 ciphertext string.

        encrypt_value(142.5)  →  "gAAAAABl..."
    """
    fernet    = _get_fernet()
    plaintext = str(value).encode("utf-8")
    return fernet.encrypt(plaintext).decode("utf-8")


def decrypt_value(ciphertext: str) -> Optional[float]:
    """
    Decrypt a ciphertext string back to a float.
    Returns None on failure (tampered data, wrong key) rather than crashing.

        decrypt_value("gAAAAABl...")  →  142.5
    """
    if ciphertext is None:
        return None
    fernet = _get_fernet()
    try:
        plaintext = fernet.decrypt(ciphertext.encode("utf-8"))
        return float(plaintext.decode("utf-8"))
    except Exception:
        logger.error(
            "[crypto] Decryption failed for ciphertext starting with: %s",
            ciphertext[:20],
        )
        return None


# ── SQLAlchemy TypeDecorator ───────────────────────────────────────────────────

class EncryptedFloat(TypeDecorator):
    """
    A SQLAlchemy column type that transparently encrypts floats at rest.

    - Database storage type: Text  (holds the Fernet ciphertext)
    - Python-side type:      float (the column behaves like a Float)

    The TypeDecorator hooks into SQLAlchemy's bind/result processor pipeline:
        process_bind_param   — called when WRITING to the DB  (float → str)
        process_result_value — called when READING from the DB (str → float)
    """

    impl            = String    # underlying DB column type
    cache_ok        = True      # safe to cache the compiled column expression

    def process_bind_param(self, value, dialect) -> Optional[str]:
        """Encrypt before writing to the database."""
        if value is None:
            return None
        return encrypt_value(float(value))

    def process_result_value(self, value, dialect) -> Optional[float]:
        """Decrypt after reading from the database."""
        if value is None:
            return None
        return decrypt_value(value)


# ── CLI utility ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--generate-key" in sys.argv:
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        print(f"Generated key (add to .env as FIELD_ENCRYPTION_KEY):\n{key}")
        sys.exit(0)

    # Self-test
    test_values = [142.5, 0.0, -1.5, 98.6, 120.0, None]
    print("EncryptedFloat self-test:")
    for v in test_values:
        if v is None:
            ct = None
            rt = None
        else:
            ct = encrypt_value(v)
            rt = decrypt_value(ct)
        print(f"  {v!r:>8} → ciphertext[:20]={str(ct)[:20]!r:>22}  → decrypted={rt!r}")
        if v is not None:
            assert rt == v, f"Round-trip failed for {v}"
    print("All assertions passed.")
