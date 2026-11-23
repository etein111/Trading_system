"""Gmail SMTP delivery for local trading reports."""
from __future__ import annotations
import os
import subprocess
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable, Optional

def _keychain_secret(service: str, account: Optional[str] = None) -> Optional[str]:
    command = ["security", "find-generic-password", "-s", service, "-w"]
    if account:
        command.extend(["-a", account])
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None

def gmail_credentials() -> tuple[str, str]:
    user = os.environ.get("TRADING_GMAIL_USER") or os.environ.get("GMAIL_USER")
    password = os.environ.get("TRADING_GMAIL_APP_PASSWORD") or os.environ.get("GMAIL_APP_PASSWORD")
    if not user:
        raise RuntimeError("set TRADING_GMAIL_USER or GMAIL_USER")
    if not password:
        password = _keychain_secret("trading-system-gmail-app-password", account=user)
    if not password:
        raise RuntimeError("set TRADING_GMAIL_APP_PASSWORD or store a Gmail app password in macOS Keychain service trading-system-gmail-app-password")
    return user, password

def send_gmail_report(subject: str, body: str, recipients: Iterable[str], attachments: Optional[Iterable[str]] = None) -> None:
    user, password = gmail_credentials()
    message = EmailMessage()
    message["From"] = user
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject
    message.set_content(body)
    for path in attachments or []:
        location = Path(path)
        message.add_attachment(location.read_bytes(), maintype="application", subtype="octet-stream", filename=location.name)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
        smtp.login(user, password)
        smtp.send_message(message)
