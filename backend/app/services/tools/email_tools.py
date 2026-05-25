from __future__ import annotations

import imaplib
import smtplib
import email as email_lib
import os

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.header import decode_header
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from pydantic import BaseModel, Field


# =========================================================
# CONFIGURATION
# Read from environment variables. Set these in your .env:
#
#   EMAIL_ADDRESS      your full email address
#   EMAIL_PASSWORD     your password or app-specific password
#   SMTP_HOST          e.g. smtp.gmail.com
#   SMTP_PORT          e.g. 587
#   IMAP_HOST          e.g. imap.gmail.com
#   IMAP_PORT          e.g. 993
# =========================================================

def _cfg(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# =========================================================
# SCHEMAS
# =========================================================

class SendEmailSchema(BaseModel):
    to: List[str] = Field(..., description="List of recipient email addresses.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Email body text (plain text or HTML).")
    cc: Optional[List[str]] = Field(None, description="Optional CC recipients.")
    bcc: Optional[List[str]] = Field(None, description="Optional BCC recipients.")
    attachment_paths: Optional[List[str]] = Field(None, description="Optional list of file paths to attach.")
    html: bool = Field(False, description="If True, body is treated as HTML.")


class ReadEmailSchema(BaseModel):
    message_id: str = Field(..., description="The Message-ID or UID of the email to read.")
    mailbox: str = Field("INBOX", description="Mailbox folder to look in.")


class ListEmailsSchema(BaseModel):
    mailbox: str = Field("INBOX", description="Mailbox folder to list.")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of emails to return.")
    search: str = Field("ALL", description="IMAP search criteria (e.g. 'UNSEEN', 'FROM someone@example.com', 'SUBJECT keyword').")


# =========================================================
# HELPERS
# =========================================================

def _decode_header_value(value: str) -> str:
    """Decode encoded email header values (e.g. =?utf-8?b?...?=)."""
    parts = decode_header(value)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return " ".join(decoded)


def _get_body(msg) -> str:
    """Extract plain text body from an email.Message object."""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", ""))
            if ctype == "text/plain" and "attachment" not in disp:
                charset = part.get_content_charset() or "utf-8"
                return part.get_payload(decode=True).decode(
                    charset, errors="replace"
                )
    else:
        charset = msg.get_content_charset() or "utf-8"
        return msg.get_payload(decode=True).decode(charset, errors="replace")
    return ""


def _imap_connect() -> imaplib.IMAP4_SSL:
    host = _cfg("IMAP_HOST", "imap.gmail.com")
    port = int(_cfg("IMAP_PORT", "993"))
    user = _cfg("EMAIL_ADDRESS")
    password = _cfg("EMAIL_PASSWORD")

    if not user or not password:
        raise EnvironmentError(
            "EMAIL_ADDRESS and EMAIL_PASSWORD environment variables must be set."
        )

    imap = imaplib.IMAP4_SSL(host, port)
    imap.login(user, password)
    return imap


# =========================================================
# IMPLEMENTATIONS
# =========================================================

def send_email(
    to: List[str],
    subject: str,
    body: str,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    attachment_paths: Optional[List[str]] = None,
    html: bool = False,
) -> str:
    """
    Send an email via SMTP.
    Supports plain text and HTML bodies, CC/BCC, and file attachments.
    """

    sender = _cfg("EMAIL_ADDRESS")
    password = _cfg("EMAIL_PASSWORD")
    smtp_host = _cfg("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(_cfg("SMTP_PORT", "587"))

    if not sender or not password:
        return (
            "ERROR: EMAIL_ADDRESS and EMAIL_PASSWORD "
            "environment variables must be set."
        )

    msg = MIMEMultipart("mixed")
    msg["From"] = sender
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject

    if cc:
        msg["Cc"] = ", ".join(cc)

    # Attach body
    mime_type = "html" if html else "plain"
    msg.attach(MIMEText(body, mime_type, "utf-8"))

    # Attach files
    missing_attachments = []
    if attachment_paths:
        for file_path in attachment_paths:
            p = Path(file_path).expanduser().resolve()
            if not p.exists():
                missing_attachments.append(str(p))
                continue
            part = MIMEBase("application", "octet-stream")
            with p.open("rb") as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{p.name}"',
            )
            msg.attach(part)

    # Build all recipients for sendmail
    all_recipients = list(to)
    if cc:
        all_recipients.extend(cc)
    if bcc:
        all_recipients.extend(bcc)

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, all_recipients, msg.as_string())
    except smtplib.SMTPAuthenticationError:
        return (
            "ERROR: SMTP authentication failed. "
            "Check EMAIL_ADDRESS and EMAIL_PASSWORD."
        )
    except Exception as e:
        return f"ERROR sending email: {e}"

    warnings = ""
    if missing_attachments:
        warnings = (
            f"\nWARNING: These attachments were not found and were skipped:\n"
            + "\n".join(f"  - {p}" for p in missing_attachments)
        )

    return (
        f"Sent successfully.\n"
        f"From: {sender}\n"
        f"To: {', '.join(to)}\n"
        f"CC: {', '.join(cc) if cc else 'None'}\n"
        f"Subject: {subject}\n"
        f"Timestamp: {datetime.utcnow().isoformat()}Z"
        f"{warnings}"
    )


def read_email(
    message_id: str,
    mailbox: str = "INBOX",
) -> str:
    """
    Fetch and return the full content of a specific email by UID.
    """

    try:
        imap = _imap_connect()
    except EnvironmentError as e:
        return f"ERROR: {e}"

    try:
        imap.select(mailbox, readonly=True)
        status, data = imap.uid("fetch", message_id, "(RFC822)")

        if status != "OK" or not data or data[0] is None:
            return f"ERROR: Message UID {message_id} not found in {mailbox}."

        raw = data[0][1]
        msg = email_lib.message_from_bytes(raw)

        sender = _decode_header_value(msg.get("From", "Unknown"))
        recipients = _decode_header_value(msg.get("To", "Unknown"))
        subject = _decode_header_value(msg.get("Subject", "(no subject)"))
        date_str = msg.get("Date", "")

        try:
            date = parsedate_to_datetime(date_str).isoformat()
        except Exception:
            date = date_str

        body = _get_body(msg)

        # List attachments
        attachments = []
        for part in msg.walk():
            disposition = part.get("Content-Disposition", "")
            if "attachment" in disposition:
                fname = part.get_filename()
                if fname:
                    attachments.append(_decode_header_value(fname))

        result = (
            f"From: {sender}\n"
            f"To: {recipients}\n"
            f"Subject: {subject}\n"
            f"Date: {date}\n"
            f"Attachments: {', '.join(attachments) if attachments else 'None'}\n"
            f"\n{'-' * 60}\n\n"
            f"{body.strip()}"
        )

        return result

    except Exception as e:
        return f"ERROR reading email: {e}"

    finally:
        try:
            imap.logout()
        except Exception:
            pass


def list_emails(
    mailbox: str = "INBOX",
    limit: int = 10,
    search: str = "ALL",
) -> str:
    """
    List emails in a mailbox matching an IMAP search criteria.
    Returns a summary list: UID, date, sender, subject.
    """

    try:
        imap = _imap_connect()
    except EnvironmentError as e:
        return f"ERROR: {e}"

    try:
        imap.select(mailbox, readonly=True)
        status, data = imap.uid("search", None, search)

        if status != "OK":
            return f"ERROR: IMAP search failed for criteria: {search}"

        uids = data[0].split()

        if not uids:
            return f"No emails found in {mailbox} matching: {search}"

        # Take the most recent N
        recent_uids = uids[-limit:][::-1]

        rows = []
        for uid in recent_uids:
            fetch_status, fetch_data = imap.uid(
                "fetch", uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
            )
            if fetch_status != "OK" or not fetch_data or fetch_data[0] is None:
                continue

            raw_headers = fetch_data[0][1]
            msg = email_lib.message_from_bytes(raw_headers)

            sender = _decode_header_value(msg.get("From", "Unknown"))
            subject = _decode_header_value(msg.get("Subject", "(no subject)"))
            date_str = msg.get("Date", "")

            try:
                date = parsedate_to_datetime(date_str).strftime("%Y-%m-%d %H:%M")
            except Exception:
                date = date_str[:20]

            rows.append(
                f"  UID {uid.decode():>6}  {date}  "
                f"{sender[:35]:<35}  {subject[:55]}"
            )

        if not rows:
            return f"No readable emails found in {mailbox}."

        header = (
            f"Mailbox: {mailbox} | Search: {search} | "
            f"Showing {len(rows)} of {len(uids)} total\n"
            + "-" * 100
        )

        return header + "\n" + "\n".join(rows)

    except Exception as e:
        return f"ERROR listing emails: {e}"

    finally:
        try:
            imap.logout()
        except Exception:
            pass


# =========================================================
# TOOL REGISTRY ENTRIES
# =========================================================

EMAIL_TOOLS = {
    "send_email": {
        "func": send_email,
        "schema": SendEmailSchema,
        "description": (
            "Send an email via SMTP. Supports plain text and HTML bodies, "
            "CC/BCC, and file attachments. Requires EMAIL_ADDRESS, "
            "EMAIL_PASSWORD, SMTP_HOST, SMTP_PORT env vars."
        ),
    },
    "read_email": {
        "func": read_email,
        "schema": ReadEmailSchema,
        "description": (
            "Read the full content of an email by its UID. "
            "Requires IMAP credentials in environment variables."
        ),
    },
    "list_emails": {
        "func": list_emails,
        "schema": ListEmailsSchema,
        "description": (
            "List emails in a mailbox. Supports IMAP search filters "
            "(UNSEEN, FROM, SUBJECT, etc). Returns UID, date, sender, subject."
        ),
    },
}
