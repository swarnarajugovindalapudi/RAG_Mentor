"""Google and SMTP Email OTP authentication for the RAGMentor Streamlit app.

Google sign-in uses Firebase Web SDK and server-side ID-token verification.
Email sign-in uses a short-lived, rate-limited SMTP OTP.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import ssl
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv(override=False)

_COMPONENT_DIR = Path(__file__).resolve().parent / "firebase_auth_component"
_firebase_auth_component = components.declare_component(
    "ragmentor_firebase_auth",
    path=str(_COMPONENT_DIR),
)

OTP_TTL_SECONDS = 300
OTP_RESEND_WAIT_SECONDS = 60
OTP_MAX_ATTEMPTS = 5


def _firebase_config() -> dict:
    """Read the public Firebase Web App configuration from environment."""
    keys = {
        "apiKey": "FIREBASE_API_KEY",
        "authDomain": "FIREBASE_AUTH_DOMAIN",
        "projectId": "FIREBASE_PROJECT_ID",
        "storageBucket": "FIREBASE_STORAGE_BUCKET",
        "messagingSenderId": "FIREBASE_MESSAGING_SENDER_ID",
        "appId": "FIREBASE_APP_ID",
    }
    config = {
        field: (os.getenv(env_name) or "").strip()
        for field, env_name in keys.items()
    }
    missing = [env_name for field, env_name in keys.items() if not config[field]]
    if missing:
        raise RuntimeError(
            "Missing Firebase configuration in .env: " + ", ".join(missing)
        )
    return config


def _verify_firebase_id_token(id_token: str, api_key: str) -> dict:
    """Ask Firebase to validate the ID token and return the Firebase user."""
    endpoint = (
        "https://identitytoolkit.googleapis.com/v1/accounts:lookup?key="
        + api_key
    )
    payload = json.dumps({"idToken": id_token}).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            error_data = json.loads(exc.read().decode("utf-8"))
            message = error_data.get("error", {}).get(
                "message", "Token verification failed"
            )
        except Exception:
            message = "Token verification failed"
        raise ValueError(message) from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError(
            f"Could not contact Firebase to verify login: {exc}"
        ) from exc

    users = data.get("users") or []
    if not users:
        raise ValueError("Firebase did not return a valid user for this login.")
    return users[0]


def _complete_login(user_info: dict, id_token: str) -> None:
    """Store only a Firebase-verified identity in Streamlit session state."""
    display_name = (
        user_info.get("displayName")
        or user_info.get("email")
        or user_info.get("phoneNumber")
        or "RAGMentor User"
    )
    verified_user = {
        "uid": user_info.get("localId"),
        "name": display_name,
        "email": user_info.get("email"),
        "phone": user_info.get("phoneNumber"),
        "method": "google",
        "email_verified": bool(user_info.get("emailVerified", False)),
    }
    st.session_state["authenticated_user"] = verified_user
    st.session_state["authenticated"] = True
    st.session_state["user_label"] = display_name
    st.session_state["firebase_id_token"] = id_token
    st.session_state["auth_step"] = "main"
    st.session_state.pop("auth_error", None)


class _IPv4SMTP(smtplib.SMTP):
    """SMTP client that resolves and connects using IPv4 addresses only."""

    def _get_socket(self, host, port, timeout):
        try:
            addresses = socket.getaddrinfo(
                host, port, family=socket.AF_INET, type=socket.SOCK_STREAM
            )
        except socket.gaierror as exc:
            raise OSError(
                f"Could not resolve {host} to an IPv4 address. "
                "Check DNS/network settings."
            ) from exc

        last_error = None
        for family, socktype, proto, _canonname, sockaddr in addresses:
            sock = socket.socket(family, socktype, proto)
            if timeout is not smtplib.socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            try:
                sock.connect(sockaddr)
                return sock
            except OSError as exc:
                last_error = exc
                sock.close()

        if last_error:
            raise last_error
        raise OSError(f"No IPv4 address was returned for {host}.")


class _IPv4SMTP_SSL(smtplib.SMTP_SSL):
    """SMTP-over-SSL client with explicit IPv4 connection and TLS hostname."""

    def _get_socket(self, host, port, timeout):
        try:
            addresses = socket.getaddrinfo(
                host, port, family=socket.AF_INET, type=socket.SOCK_STREAM
            )
        except socket.gaierror as exc:
            raise OSError(
                f"Could not resolve {host} to an IPv4 address. "
                "Check DNS/network settings."
            ) from exc

        last_error = None
        for family, socktype, proto, _canonname, sockaddr in addresses:
            raw_sock = socket.socket(family, socktype, proto)
            if timeout is not smtplib.socket._GLOBAL_DEFAULT_TIMEOUT:
                raw_sock.settimeout(timeout)
            try:
                raw_sock.connect(sockaddr)
                return self.context.wrap_socket(raw_sock, server_hostname=host)
            except OSError as exc:
                last_error = exc
                raw_sock.close()

        if last_error:
            raise last_error
        raise OSError(f"No IPv4 address was returned for {host}.")


def _send_email_otp(recipient: str, otp: str) -> None:
    """Send an OTP through Gmail SMTP, preferring IPv4 to avoid IPv6 DNS issues."""
    host = (os.getenv("SMTP_HOST") or "smtp.gmail.com").strip()
    port = int(os.getenv("SMTP_PORT", "587"))
    username = (os.getenv("SMTP_USER") or "").strip()
    password = (os.getenv("SMTP_APP_PASSWORD") or "").replace(" ", "").strip()
    sender = (os.getenv("SMTP_FROM") or username).strip()

    missing = [
        name for name, value in (
            ("SMTP_USER", username),
            ("SMTP_APP_PASSWORD", password),
            ("SMTP_FROM or SMTP_USER", sender),
        ) if not value
    ]
    if missing:
        raise RuntimeError("Email OTP setup missing: " + ", ".join(missing))

    if port not in (465, 587):
        raise RuntimeError(
            "Unsupported SMTP_PORT. Use 587 for STARTTLS or 465 for SSL."
        )

    message = EmailMessage()
    message["Subject"] = "Your RAGMentor verification code"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(
        f"""Your RAGMentor verification code is: {otp}

This code expires in 5 minutes. Do not share it with anyone.

If you did not request this code, you can ignore this email.
"""
    )

    try:
        if port == 465:
            with _IPv4SMTP_SSL(
                host, port, timeout=30,
                context=ssl.create_default_context()
            ) as server:
                server.ehlo()
                server.login(username, password)
                server.send_message(message)
        else:
            with _IPv4SMTP(host, port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                server.login(username, password)
                server.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        raise RuntimeError(
            f"Gmail SMTP failed ({type(exc).__name__}): {exc}. "
            "Check SMTP_USER, Gmail App Password, SMTP_HOST/PORT, and network access."
        ) from exc


def _otp_digest(email: str, otp: str) -> str:
    secret = os.getenv("OTP_HASH_SECRET") or os.getenv("SMTP_APP_PASSWORD") or "ragmentor-session-otp"
    value = f"{email.lower()}:{otp}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), value, hashlib.sha256).hexdigest()


def _render_email_otp_login() -> None:
    """Render the email OTP option and create a session login after verification."""
    st.markdown(
        """
        <div style="text-align:center;color:#dbeafe;margin:1.2rem 0 .7rem;">
          <span style="opacity:.85;">────────── or continue with email ──────────</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("### Continue with Email")
        st.caption("Enter your Gmail address. We will send you a 6-digit OTP.")

        email = st.text_input(
            "Gmail / Email address",
            placeholder="yourname@gmail.com",
            key="ragmentor_email_otp_address",
        ).strip().lower()

        send_col, _ = st.columns([1, 1])
        with send_col:
            if st.button(
                "Send Email OTP",
                key="ragmentor_send_email_otp",
                use_container_width=True,
            ):
                if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
                    st.error("Please enter a valid email address.")
                else:
                    now = time.time()
                    last_sent = float(
                        st.session_state.get("email_otp_last_sent_at", 0)
                    )
                    previous_email = st.session_state.get("email_otp_email")
                    if previous_email == email and now - last_sent < OTP_RESEND_WAIT_SECONDS:
                        remaining = int(OTP_RESEND_WAIT_SECONDS - (now - last_sent))
                        st.warning(f"Please wait {remaining} seconds before requesting another OTP.")
                    else:
                        otp = f"{secrets.randbelow(900000) + 100000:06d}"
                        try:
                            _send_email_otp(email, otp)
                            st.session_state["email_otp_email"] = email
                            st.session_state["email_otp_digest"] = _otp_digest(email, otp)
                            st.session_state["email_otp_expires_at"] = now + OTP_TTL_SECONDS
                            st.session_state["email_otp_last_sent_at"] = now
                            st.session_state["email_otp_attempts"] = 0
                            st.session_state["email_otp_verified"] = False
                            st.success(f"OTP sent to {email}. Check your inbox and spam folder.")
                        except Exception as exc:
                            st.error(f"Could not send email OTP: {exc}")

        entered_otp = st.text_input(
            "Enter 6-digit OTP",
            max_chars=6,
            placeholder="123456",
            key="ragmentor_email_otp_code",
        )

        if st.button(
            "Verify OTP & Continue",
            type="primary",
            key="ragmentor_verify_email_otp",
            use_container_width=True,
        ):
            stored_email = st.session_state.get("email_otp_email")
            stored_digest = st.session_state.get("email_otp_digest")
            expires_at = float(st.session_state.get("email_otp_expires_at", 0))
            attempts = int(st.session_state.get("email_otp_attempts", 0))

            if not stored_digest or not stored_email:
                st.warning("Please send an OTP to your email first.")
            elif email != stored_email:
                st.error("The email address does not match the address used to send the OTP.")
            elif time.time() > expires_at:
                st.session_state.pop("email_otp_digest", None)
                st.error("OTP expired. Please request a new OTP.")
            elif attempts >= OTP_MAX_ATTEMPTS:
                st.session_state.pop("email_otp_digest", None)
                st.error("Too many incorrect attempts. Please request a new OTP.")
            elif not re.fullmatch(r"\d{6}", (entered_otp or "").strip()):
                st.error("Enter the 6-digit OTP sent to your email.")
            else:
                candidate = _otp_digest(email, entered_otp.strip())
                if hmac.compare_digest(candidate, stored_digest):
                    user = {
                        "uid": f"email:{email}",
                        "name": email.split("@", 1)[0],
                        "email": email,
                        "phone": None,
                        "method": "email_otp",
                        "email_verified": True,
                    }
                    st.session_state["authenticated_user"] = user
                    st.session_state["authenticated"] = True
                    st.session_state["user_label"] = user["name"]
                    st.session_state["auth_step"] = "main"
                    st.session_state.pop("email_otp_digest", None)
                    st.session_state.pop("auth_error", None)
                    st.success("Email verified. Opening your dashboard...")
                    st.rerun()
                else:
                    st.session_state["email_otp_attempts"] = attempts + 1
                    remaining = OTP_MAX_ATTEMPTS - attempts - 1
                    st.error(f"Incorrect OTP. {remaining} attempts remaining.")


def get_logged_in_user() -> dict | None:
    """Return the authenticated user currently logged in to this session."""
    return st.session_state.get("authenticated_user")


def logout_auth_flow() -> None:
    """Clear authentication state and force a fresh browser auth component."""
    for key in (
        "authenticated_user",
        "authenticated",
        "user_label",
        "firebase_id_token",
        "auth_step",
        "auth_error",
        "email_otp_email",
        "email_otp_digest",
        "email_otp_expires_at",
        "email_otp_last_sent_at",
        "email_otp_attempts",
        "email_otp_verified",
        "email_component_status",
        "last_email_component_action",
        "ragmentor_email_otp_address",
        "ragmentor_email_otp_code",
    ):
        st.session_state.pop(key, None)
    st.session_state["firebase_logout_version"] = (
        int(st.session_state.get("firebase_logout_version", 0)) + 1
    )



def _handle_email_component_action(result: dict) -> None:
    """Process email OTP actions sent by the embedded login component."""
    action = result.get("action")
    email = (result.get("email") or "").strip().lower()

    if action == "send_email_otp":
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            st.session_state["email_component_status"] = {
                "message": "Please enter a valid email address.", "type": "error"
            }
            return

        now = time.time()
        last_sent = float(st.session_state.get("email_otp_last_sent_at", 0))
        previous_email = st.session_state.get("email_otp_email")
        if previous_email == email and now - last_sent < OTP_RESEND_WAIT_SECONDS:
            remaining = int(OTP_RESEND_WAIT_SECONDS - (now - last_sent))
            st.session_state["email_component_status"] = {
                "message": f"Please wait {remaining} seconds before requesting another OTP.",
                "type": "error"
            }
            return

        otp = f"{secrets.randbelow(900000) + 100000:06d}"
        try:
            _send_email_otp(email, otp)
            st.session_state["email_otp_email"] = email
            st.session_state["email_otp_digest"] = _otp_digest(email, otp)
            st.session_state["email_otp_expires_at"] = now + OTP_TTL_SECONDS
            st.session_state["email_otp_last_sent_at"] = now
            st.session_state["email_otp_attempts"] = 0
            st.session_state["email_component_status"] = {
                "message": f"OTP sent to {email}. Check your inbox and spam folder.",
                "type": "success"
            }
        except Exception as exc:
            st.session_state["email_component_status"] = {
                "message": f"Could not send email OTP: {exc}", "type": "error"
            }

    elif action == "verify_email_otp":
        entered = (result.get("otp") or "").strip()
        stored_email = st.session_state.get("email_otp_email")
        stored_digest = st.session_state.get("email_otp_digest")
        expires_at = float(st.session_state.get("email_otp_expires_at", 0))
        attempts = int(st.session_state.get("email_otp_attempts", 0))

        if not stored_digest or not stored_email:
            message = "Please request an OTP first."
        elif email != stored_email:
            message = "The email address does not match the OTP recipient."
        elif time.time() > expires_at:
            st.session_state.pop("email_otp_digest", None)
            message = "OTP expired. Please request a new OTP."
        elif attempts >= OTP_MAX_ATTEMPTS:
            st.session_state.pop("email_otp_digest", None)
            message = "Too many incorrect attempts. Please request a new OTP."
        elif not re.fullmatch(r"\d{6}", entered):
            message = "Enter the 6-digit OTP sent to your email."
        elif hmac.compare_digest(_otp_digest(email, entered), stored_digest):
            user = {
                "uid": f"email:{email}",
                "name": email.split("@", 1)[0],
                "email": email,
                "phone": None,
                "method": "email_otp",
                "email_verified": True,
            }
            st.session_state["authenticated_user"] = user
            st.session_state["authenticated"] = True
            st.session_state["user_label"] = user["name"]
            st.session_state["auth_step"] = "main"
            st.session_state.pop("email_otp_digest", None)
            st.session_state.pop("auth_error", None)
            st.rerun()
            return
        else:
            st.session_state["email_otp_attempts"] = attempts + 1
            message = (
                f"Incorrect OTP. {OTP_MAX_ATTEMPTS - attempts - 1} attempts remaining."
            )

        st.session_state["email_component_status"] = {
            "message": message, "type": "error"
        }


def require_login(google_accounts=None) -> bool:
    """Block the app until Firebase Google/phone or SMTP email OTP succeeds."""
    if st.session_state.get("authenticated_user"):
        return True

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background:
              radial-gradient(circle at 12% 12%, rgba(59,130,246,.20), transparent 32%),
              radial-gradient(circle at 88% 82%, rgba(14,165,233,.16), transparent 30%),
              linear-gradient(135deg, #07152d 0%, #102c50 52%, #0b1b35 100%);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMain"] { background: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        config = _firebase_config()
    except RuntimeError as exc:
        st.error(str(exc))
        st.info(
            "Create the .env file using the provided Firebase configuration, "
            "then restart Streamlit."
        )
        st.stop()

    st.markdown(
        "<div style='text-align:center;margin:2.2rem 0 .3rem;color:white;'>"
        "<div style='font-size:2.4rem;font-weight:800;letter-spacing:-1px;'>📘 RAGMentor</div>"
        "<div style='font-size:1rem;color:#c7d7ef;'>Learn Smarter with Your Documents</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    result = _firebase_auth_component(
        firebaseConfig=config,
        logoutVersion=int(st.session_state.get("firebase_logout_version", 0)),
        emailStatus=st.session_state.get("email_component_status", {}),
        key=(
            "ragmentor_firebase_auth_"
            f"{int(st.session_state.get('firebase_logout_version', 0))}"
        ),
        default=None,
    )

    if isinstance(result, dict) and result.get("action"):
        # Streamlit returns the component value on rerun. Process each click once.
        action_fingerprint = json.dumps(result, sort_keys=True)
        if action_fingerprint != st.session_state.get("last_email_component_action"):
            st.session_state["last_email_component_action"] = action_fingerprint
            _handle_email_component_action(result)
            # The component was rendered before the server action ran.
            # Rerun so its iframe receives the updated success/error status.
            st.rerun()

    if isinstance(result, dict) and result.get("idToken"):
        try:
            firebase_user = _verify_firebase_id_token(
                result["idToken"], config["apiKey"]
            )
            _complete_login(firebase_user, result["idToken"])
            st.rerun()
        except ValueError as exc:
            st.session_state["auth_error"] = str(exc)

    if st.session_state.get("auth_error"):
        st.error(st.session_state["auth_error"])


    st.markdown(
        "<div style='text-align:center;color:#c7d7ef;font-size:.82rem;margin-top:.8rem;'>"
        "Google sign-in powered by Firebase · Email OTP powered by SMTP"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()
    return False
