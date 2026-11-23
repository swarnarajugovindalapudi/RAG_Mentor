import io
import random
import streamlit as st
import speech_recognition as sr


def init_auth_voice_state():
    """Initialize session state keys for login, temporary chat, and voice input."""
    defaults = {
        "authenticated": False,
        "auth_mode": None,  # "phone_otp", "google", "demo"
        "user_label": None,
        "otp_code": None,
        "otp_sent": False,
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
    st.session_state.otp_sent = False
    st.session_state.otp_code = None


def logout_user():
    """Reset authentication and temporary session state."""
    st.session_state.authenticated = False
    st.session_state.auth_mode = None
    st.session_state.user_label = None
    st.session_state.otp_sent = False
    st.session_state.otp_code = None
    st.session_state.temp_chat = False
    st.session_state.voice_question = ""


def render_login_screen():
    """Render ChatGPT-style multi-option login (Phone/OTP, Google/Gmail, Demo)."""
    init_auth_voice_state()

    st.markdown("### Welcome to RAGMentor")
    st.caption("Sign in to chat with your PDFs, use voice questions, and save your study workspace.")

    tab_demo, tab_google, tab_phone = st.tabs(
        ["Quick Demo", "Continue with Google / Gmail", "Phone + OTP"]
    )

    with tab_demo:
        st.write("Jump right in without creating an account.")
        demo_name = st.text_input("Display Name (Optional)", value="Student Guest", key="demo_name_input")
        if st.button("Enter Demo Workspace", type="primary", use_container_width=True):
            _login_success("demo", demo_name.strip() or "Student Guest")
            st.rerun()

    with tab_google:
        email = st.text_input(
            "Enter Gmail / Google Address",
            placeholder="student@gmail.com",
            key="google_email_input",
        )
        if st.button("Continue with Google", type="primary", use_container_width=True):
            cleaned_email = (email or "").strip()
            if "@" not in cleaned_email or "." not in cleaned_email.split("@")[-1]:
                st.error("Please enter a valid Gmail or Google email address.")
            else:
                _login_success("google", cleaned_email)
                st.rerun()

    with tab_phone:
        phone = st.text_input(
            "Mobile Number",
            placeholder="+91 98765 43210",
            key="phone_number_input",
        )
        col_send, col_verify = st.columns([1, 1])

        with col_send:
            if st.button("Send OTP", use_container_width=True):
                digits = "".join(ch for ch in (phone or "") if ch.isdigit())
                if len(digits) < 10:
                    st.error("Please enter a valid 10-digit mobile number.")
                else:
                    otp = f"{random.randint(100000, 999999)}"
                    st.session_state.otp_code = otp
                    st.session_state.otp_sent = True
                    st.info(f"Demo OTP sent: **{otp}**")

        entered_otp = st.text_input("Enter 6-Digit OTP", max_chars=6, key="otp_input")
        with col_verify:
            if st.button("Verify & Login", type="primary", use_container_width=True):
                if not st.session_state.otp_sent:
                    st.warning("Click 'Send OTP' first.")
                elif (entered_otp or "").strip() == st.session_state.otp_code:
                    _login_success("phone_otp", (phone or "").strip())
                    st.rerun()
                else:
                    st.error("Invalid OTP. Please check and try again.")


def render_temp_chat_toggle():
    """Render Temporary Chat mode toggle."""
    init_auth_voice_state()
    temp_enabled = st.toggle(
        "Temporary Chat",
        value=st.session_state.temp_chat,
        help="When enabled, chat history is not saved persistently.",
    )
    st.session_state.temp_chat = temp_enabled
    if temp_enabled:
        st.caption("Temporary Chat active: session messages won't be saved to history.")


def transcribe_audio_bytes(audio_file, language="en-IN"):
    """
    Convert uploaded/recorded audio bytes from st.audio_input into text
    using SpeechRecognition (Google Web Speech API).
    """
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
        return {"text": "", "error": "Could not understand the speech clearly. Please try again."}
    except sr.RequestError as exc:
        return {"text": "", "error": f"Speech recognition service error: {exc}"}
    except Exception as exc:
        return {"text": "", "error": f"Audio processing failed: {exc}"}