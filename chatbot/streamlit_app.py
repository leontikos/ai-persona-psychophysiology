"""Streamlit chat interface used to deliver the two AI persona conditions.

The two conditions differ only in the system prompt loaded from `personas/`.
Model, temperature, context window, interface, and streaming are identical.
Every conversation is written to a JSON log with per-message roles, contents,
start/end timestamps and token counts.
"""

import datetime
import json
import os
import time
from pathlib import Path

import streamlit as st
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL = "gpt-4o"
MAX_TOKENS = 64000  # Context budget; oldest exchanges are dropped beyond this.
TEMPERATURE = 0.0

PERSONA_DIR = Path(__file__).parent / "personas"
LOG_DIR = Path(os.environ.get("CONVERSATION_LOG_DIR", Path(__file__).parent / "conversation_logs"))

# Condition 1 = Servant Leader, condition 2 = Dark Triad. The operator selects
# the number only, so the label does not reveal the manipulation to participants.
PERSONA_FILES = {"1": "servant_leader.txt", "2": "dark_triad.txt"}

GREETING = "How may I help you?"


def load_system_prompt(chatbot_version):
    """Return the verbatim system prompt for the given condition."""
    return (PERSONA_DIR / PERSONA_FILES[chatbot_version]).read_text(encoding="utf-8").rstrip("\n")


def count_tokens(text):
    return len(tiktoken.encoding_for_model(MODEL).encode(text))


def count_messages_tokens(messages):
    """Approximate the token count of a full API message list."""
    encoding = tiktoken.encoding_for_model(MODEL)
    total = 3  # End-of-sequence overhead.
    for message in messages:
        total += 4  # Per-message role overhead.
        if message.get("content"):
            total += len(encoding.encode(message["content"]))
    return total


def save_conversation(messages, conversation_id, user_id, chatbot_version):
    """Write the running conversation and its metadata to a JSON log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    conversation_data = {
        "metadata": {
            "user_id": user_id,
            "chatbot_version": chatbot_version,
            "total_messages": len(messages),
            "total_tokens": count_messages_tokens(
                [{"role": m["role"], "content": m["content"]} for m in messages]
            ),
            "model": MODEL,
            "last_updated": datetime.datetime.now().isoformat(),
        },
        "messages": messages,
    }

    with open(LOG_DIR / conversation_id, "w", encoding="utf-8") as handle:
        json.dump(conversation_data, handle, indent=2, ensure_ascii=False)


def generate_response(messages, chatbot_version):
    """Stream a completion, truncating the oldest exchanges to fit MAX_TOKENS."""
    api_messages = [{"role": "system", "content": load_system_prompt(chatbot_version)}]
    api_messages += [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]

    while count_messages_tokens(api_messages) > MAX_TOKENS:
        if len(api_messages) > 2:
            api_messages.pop(1)  # Drop the oldest turn, keeping the system prompt.
        else:
            content = api_messages[-1]["content"]
            api_messages[-1]["content"] = content[: len(content) // 2]

    return client.chat.completions.create(
        model=MODEL,
        messages=api_messages,
        temperature=TEMPERATURE,
        stream=True,
    )


st.set_page_config(page_title="💬 Research Chatbot")

# Strip Streamlit branding so participants see an unmarked research tool.
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display: none !important;}
    div[data-testid="stAppDeployButton"] {display: none !important;}
    button[data-testid="stBaseButton-header"] {display: none !important;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    .stAppToolbar {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("user_info_collected", False)
st.session_state.setdefault("user_id", "")
st.session_state.setdefault("chatbot_version", "1")

if not st.session_state.user_info_collected:
    st.title("Welcome to Research Chatbot")
    st.subheader("Please provide the following information to get started:")

    user_id = st.text_input("User ID:", key="user_id_input")
    chatbot_version = st.radio("Chatbot Version:", ["1", "2"], key="chatbot_version_input")

    if st.button("Start Chatting"):
        if user_id:
            st.session_state.user_id = user_id
            st.session_state.chatbot_version = chatbot_version
            st.session_state.user_info_collected = True
            st.rerun()
        else:
            st.error("Please enter a User ID to continue.")
else:
    if "conversation_id" not in st.session_state:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.conversation_id = f"conversation_{timestamp}.json"

    if "messages" not in st.session_state:
        now = datetime.datetime.now().isoformat()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": GREETING,
                "timestamp_start": now,
                "timestamp_end": now,
                "tokens": count_tokens(GREETING),
            }
        ]
        save_conversation(
            st.session_state.messages,
            st.session_state.conversation_id,
            st.session_state.user_id,
            st.session_state.chatbot_version,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        # Participant messages carry no typing duration, so start == end.
        timestamp = datetime.datetime.now().isoformat()
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "timestamp_start": timestamp,
                "timestamp_end": timestamp,
                "tokens": count_tokens(prompt),
            }
        )

        with st.chat_message("user"):
            st.write(prompt)

        save_conversation(
            st.session_state.messages,
            st.session_state.conversation_id,
            st.session_state.user_id,
            st.session_state.chatbot_version,
        )

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            timestamp_start = datetime.datetime.now().isoformat()
            placeholder = st.empty()
            full_response = ""

            for chunk in generate_response(
                st.session_state.messages, st.session_state.chatbot_version
            ):
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    full_response += delta.content
                    placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)  # Paces the stream into a typing effect.

            placeholder.markdown(full_response)
            timestamp_end = datetime.datetime.now().isoformat()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "timestamp_start": timestamp_start,
                "timestamp_end": timestamp_end,
                "tokens": count_tokens(full_response),
            }
        )

        save_conversation(
            st.session_state.messages,
            st.session_state.conversation_id,
            st.session_state.user_id,
            st.session_state.chatbot_version,
        )
