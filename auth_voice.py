import io
import random
import smtplib
import os
from email.message import EmailMessage
import streamlit as st
import speech_recognition as sr


def init_auth_voice_state():
    """Initialize session state keys for login, temporary chat, and voice input."""
    defaults = {
        "authenticated": False,
        "auth_mode": None,  # "phone_otp", "email_otp", "google", "demo"
        "user_label": None,
        "phone_otp_code": None,
        "phone_otp_sent": False,
        "email_otp_code": None,
        "email_otp_sent": False,
        "temp_chat": False,
        "voice_question": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _login_success(mode, label):
    """Set authenticated session state."""
    st.session_state.authenticated = True
    st.session_state.auth_mode = mode
    st.session_state.user_label = label
    st.session_state.phone_otp_sent = False
    st.session_state.phone_otp_code = None
    st.session_state.email_otp_sent = False
    st.session_state.email_otp_code = None


def logout_user():
    """Reset authentication and temporary session state."""
    st.session_state.authenticated = False
    st.session_state.auth_mode = None
    st.session_state.user_label = None
    st.session_state.phone_otp_sent = False
    st.session_state.phone_otp_code = None
    st.session_state.email_otp_sent = False
    st.session_state.email_otp_code = None
    st.session_state.temp_chat = False
    st.session_state.voice_question = ""


def _send_email_otp(recipient_email, otp_code):
    """Send OTP via SMTP if configured; otherwise return False so UI shows demo/local OTP."""
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_pass = os.getenv("SMTP_PASS", "").strip()

    if not (smtp_host and smtp_user and smtp_pass):
        return False

    msg = EmailMessage()
    msg["Subject"] = "RAGMentor Login Verification OTP"
    msg["From"] = smtp_user
    msg["To"] = recipient_email
    msg.set_content(f"Your RAGMentor 6-digit verification OTP is: {otp_code}")

    with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
    return True


def render_login_screen():
    """Render login screen with Phone + OTP, Email + OTP / Google, and Quick Demo."""
    init_auth_voice_state()

    st.markdown(
        """
        <style>
        .login-wrap {
            max-width: 540px;
            margin: 2rem auto 0 auto;
            padding: 28px 32px;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            background: #ffffff;
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
        }
        .login-brand {
            font-size: 26px;
            font-weight: 700;
            color: #1d4ed8;
            margin-bottom: 4px;
        }
        .login-sub {
            color: #64748b;
            font-size: 14px;
            margin-bottom: 18px;
        }
        </style>
        <div class="login-wrap">
            <div class="login-brand">RAGMentor</div>
            <div class="login-sub">Sign in with Phone OTP, Email OTP / Google, or Demo workspace.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, center_col, _ = st.columns([1, 2.2, 1])
    with center_col:
        tab_phone, tab_email_google, tab_demo = st.tabs(
            ["Phone + OTP", "Email OTP / Google", "Quick Demo"]
        )

        with tab_phone:
            phone = st.text_input(
                "Mobile Number",
                placeholder="+91 98765 43210",
                key="phone_number_input",
            )
            if st.button("Send OTP", key="send_phone_otp_btn", use_container_width=True):
                digits = "".join(ch for ch in (phone or "") if ch.isdigit())
                if len(digits) < 10:
                    st.error("Please enter a valid mobile number.")
                else:
                    otp = f"{random.randint(100000, 999999)}"
                    st.session_state.phone_otp_code = otp
                    st.session_state.phone_otp_sent = True
                    st.info(f"OTP generated for {phone.strip()}: **{otp}**")

            entered_phone_otp = st.text_input(
                "Enter 6-Digit OTP",
                max_chars=6,
                key="phone_otp_input",
            )
            if st.button("Verify & Login", type="primary", key="verify_phone_otp_btn", use_container_width=True):
                if not st.session_state.phone_otp_sent:
                    st.warning("Click 'Send OTP' first.")
                elif (entered_phone_otp or "").strip() == st.session_state.phone_otp_code:
                    _login_success("phone_otp", (phone or "").strip())
                    st.rerun()
                else:
                    st.error("Invalid OTP. Please try again.")

        with tab_email_google:
            email = st.text_input(
                "Email / Gmail Address",
                placeholder="name@gmail.com",
                key="email_otp_address_input",
            )
            col_send_mail, col_google_quick = st.columns(2)

            with col_send_mail:
                if st.button("Send Email OTP", key="send_email_otp_btn", use_container_width=True):
                    cleaned = (email or "").strip()
                    if "@" not in cleaned or "." not in cleaned.split("@")[-1]:
                        st.error("Please enter a valid email address.")
                    else:
                        otp = f"{random.randint(100000, 999999)}"
                        st.session_state.email_otp_code = otp
                        st.session_state.email_otp_sent = True
                        sent_smtp = _send_email_otp(cleaned, otp)
                        if sent_smtp:
                            st.success(f"OTP sent to **{cleaned}**.")
                        else:
                            st.info(f"Email OTP for **{cleaned}**: **{otp}**")

            with col_google_quick:
                if st.button("Continue with Google", key="google_auth_btn", use_container_width=True):
                    cleaned = (email or "").strip()
                    if "@" not in cleaned or not cleaned.lower().endswith("@gmail.com"):
                        st.error("Enter a valid @gmail.com address to continue with Google.")
                    else:
                        _login_success("google", cleaned)
                        st.rerun()

            entered_email_otp = st.text_input(
                "Enter 6-Digit Email OTP",
                max_chars=6,
                key="email_otp_code_input",
            )
            if st.button("Verify Email OTP & Login", type="primary", key="verify_email_otp_btn", use_container_width=True):
                if not st.session_state.email_otp_sent:
                    st.warning("Click 'Send Email OTP' first.")
                elif (entered_email_otp or "").strip() == st.session_state.email_otp_code:
                    _login_success("email_otp", (email or "").strip())
                    st.rerun()
                else:
                    st.error("Invalid Email OTP. Please try again.")

        with tab_demo:
            demo_name = st.text_input("Display Name", value="Vyshnavi", key="demo_user_name")
            if st.button("Enter Demo Workspace", type="primary", key="demo_login_btn", use_container_width=True):
                _login_success("demo", (demo_name or "").strip() or "Vyshnavi")
                st.rerun()


def render_temp_chat_toggle():
    """Render Temporary Chat toggle in sidebar."""
    init_auth_voice_state()
    temp_enabled = st.toggle(
        "Temporary Chat",
        value=st.session_state.temp_chat,
        key="temp_chat_toggle_widget",
    )
    st.session_state.temp_chat = temp_enabled
    st.caption("Doesn't save chat history" if temp_enabled else "Saves chats to Recent Chats")


def transcribe_audio_bytes(audio_file, language="en-IN"):
    """Convert audio bytes from st.audio_input into text using SpeechRecognition."""
    if audio_file is None:
        return {"text": "", "error": "No audio recorded."}

    try:
        audio_bytes = audio_file.getvalue() if hasattr(audio_file, "getvalue") else audio_file.read()
        if not audio_bytes:
            return {"text": "", "error": "Recorded audio is empty."}

        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data, language=language)
        return {"text": text.strip(), "error": None}

    except sr.UnknownValueError:
        return {"text": "", "error": "Could not understand speech clearly. Try again."}
    except sr.RequestError as exc:
        return {"text": "", "error": f"Speech recognition error: {exc}"}
    except Exception as exc:
        return {"text": "", "error": f"Audio processing failed: {exc}"}