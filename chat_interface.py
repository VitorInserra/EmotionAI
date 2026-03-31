import streamlit as st
from typing import List, Dict

from llama_eeg import (
    delta_bucket,
    build_policy_table,
    build_base_system_message,
    build_controller_memo,
    ollama_chat,
    MODEL,
)

if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.mode = None
    st.session_state.target_arousal = None
    st.session_state.history = []
    st.session_state.base_messages = []
    st.session_state.policies = build_policy_table()


if not st.session_state.initialized:
    st.title("LLaMaEEG")
    st.subheader("Session Setup")

    mode = st.selectbox(
        "Mode",
        ["neutral", "supportive", "coach"]
    )

    target_arousal = st.slider(
        "Target Arousal (1-5)",
        min_value=1,
        max_value=5,
        value=3
    )

    if st.button("Start Session"):
        st.session_state.mode = mode
        st.session_state.target_arousal = target_arousal

        st.session_state.base_messages = [
            {
                "role": "system",
                "content": build_base_system_message(mode, target_arousal)
            }
        ]

        st.session_state.history = []
        st.session_state.initialized = True
        st.rerun()

    st.stop()

st.title("LLaMaEEG")

st.caption(
    f"Mode: {st.session_state.mode} | "
    f"Target Arousal: {st.session_state.target_arousal}"
)


current_arousal = st.slider(
    "Current Arousal (1-5)",
    min_value=1,
    max_value=5,
    value=3,
    key="arousal_slider"
)

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:

    mode = st.session_state.mode
    target = st.session_state.target_arousal

    bucket = delta_bucket(current_arousal, target)
    policy = st.session_state.policies[mode][bucket]

    controller_msg = {
        "role": "system",
        "content": build_controller_memo(
            mode=mode,
            current_arousal=current_arousal,
            target_arousal=target,
            bucket=bucket,
            policy=policy,
        ),
    }

    messages = (
        st.session_state.base_messages
        + st.session_state.history
        + [controller_msg, {"role": "user", "content": user_input}]
    )

    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = ollama_chat(
                MODEL,
                messages,
                temperature=policy.temperature,
                top_p=policy.top_p,
                repeat_penalty=policy.repeat_penalty,
                num_predict=policy.num_predict,
            )

        st.markdown(reply)

    st.session_state.history.append({"role": "assistant", "content": reply})

    st.rerun()