import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma2:2b"


def get_int_in_range(prompt: str, lo: int, hi: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
        except ValueError:
            print(f"Please enter an integer between {lo} and {hi}.")
            continue

        if lo <= val <= hi:
            return val

        print(f"Out of range. Enter an integer between {lo} and {hi}.")


def choose_mode() -> str:
    """
    Session-level mode: hard constraints + soft arousal steering.
    """
    modes = {
        "1": "neutral",
        "2": "supportive",
        "3": "coach",
    }

    print("Select mode:")
    print("  1) neutral  (flat tone, facts/checklists, no hype)")
    print("  2) supportive       (calm, reassuring, grounding)")
    print("  3) coach            (directive, action-oriented)")
    while True:
        choice = input("Mode (1-3) [default=1]: ").strip() or "1"
        if choice in modes:
            return modes[choice]
        print("Invalid choice. Enter 1, 2, or 3.")


def get_arousal_score() -> int:
    """
    Placeholder for your on-server EEG classifier.
    Replace this body with: load EEG window -> run model -> map to 1..5.
    """
    return get_int_in_range("EEG arousal (1-5): ", 1, 5)


def delta_bucket(current_arousal: int, target_arousal: int) -> str:
    """
    Buckets delta = current - target into a small set of categories.
    """
    d = current_arousal - target_arousal
    if d >= 2:
        return "HIGH"
    if d == 1:
        return "SLIGHTLY_HIGH"
    if d == 0:
        return "MATCH"
    if d == -1:
        return "SLIGHTLY_LOW"
    return "LOW"


@dataclass(frozen=True)
class ResponsePolicy:
    # Behavioral knobs
    tone: str
    format: str
    max_sentences: int
    questions_allowed: int
    avoid: Tuple[str, ...]       # phrases/styles to avoid

    temperature: float
    top_p: float
    repeat_penalty: float
    num_predict: int


def build_policy_table() -> Dict[str, Dict[str, ResponsePolicy]]:
    """
    policies[mode][bucket] -> ResponsePolicy
    """
    return {
        "neutral": {
            "HIGH": ResponsePolicy(
                tone="flat-neutral, de-escalating",
                format="checklist",
                max_sentences=5,
                questions_allowed=0,
                avoid=("hype", "reassurance fluff", "doom", "emotional adjectives"),
                temperature=0.15,
                top_p=0.80,
                repeat_penalty=1.10,
                num_predict=220,
            ),
            "SLIGHTLY_HIGH": ResponsePolicy(
                tone="flat-neutral, steady",
                format="checklist",
                max_sentences=7,
                questions_allowed=1,
                avoid=("hype", "doom", "emotional adjectives"),
                temperature=0.20,
                top_p=0.85,
                repeat_penalty=1.08,
                num_predict=260,
            ),
            "MATCH": ResponsePolicy(
                tone="flat-neutral",
                format="bullets",
                max_sentences=9,
                questions_allowed=1,
                avoid=("hype", "doom"),
                temperature=0.25,
                top_p=0.90,
                repeat_penalty=1.06,
                num_predict=320,
            ),
            "SLIGHTLY_LOW": ResponsePolicy(
                tone="flat-neutral, slightly more directive",
                format="bullets",
                max_sentences=10,
                questions_allowed=1,
                avoid=("hype", "overly motivational language"),
                temperature=0.30,
                top_p=0.92,
                repeat_penalty=1.06,
                num_predict=360,
            ),
            "LOW": ResponsePolicy(
                tone="flat-neutral, structured and directive",
                format="checklist",
                max_sentences=10,
                questions_allowed=1,
                avoid=("hype", "pep talk"),
                temperature=0.32,
                top_p=0.93,
                repeat_penalty=1.05,
                num_predict=380,
            ),
        },

        "supportive": {
            "HIGH": ResponsePolicy(
                tone="calm, grounding",
                format="bullets",
                max_sentences=6,
                questions_allowed=1,
                avoid=("judgment", "pressure", "excess detail"),
                temperature=0.20,
                top_p=0.85,
                repeat_penalty=1.08,
                num_predict=260,
            ),
            "SLIGHTLY_HIGH": ResponsePolicy(
                tone="steady, reassuring",
                format="bullets",
                max_sentences=8,
                questions_allowed=1,
                avoid=("pressure", "doom"),
                temperature=0.28,
                top_p=0.90,
                repeat_penalty=1.06,
                num_predict=320,
            ),
            "MATCH": ResponsePolicy(
                tone="warm-neutral",
                format="paragraph",
                max_sentences=10,
                questions_allowed=2,
                avoid=("doom"),
                temperature=0.35,
                top_p=0.93,
                repeat_penalty=1.05,
                num_predict=420,
            ),
            "SLIGHTLY_LOW": ResponsePolicy(
                tone="encouraging but grounded",
                format="bullets",
                max_sentences=10,
                questions_allowed=2,
                avoid=("cheesy hype",),
                temperature=0.45,
                top_p=0.95,
                repeat_penalty=1.05,
                num_predict=450,
            ),
            "LOW": ResponsePolicy(
                tone="energizing but supportive",
                format="bullets",
                max_sentences=10,
                questions_allowed=2,
                avoid=("cheesy hype", "overpromising"),
                temperature=0.55,
                top_p=0.96,
                repeat_penalty=1.04,
                num_predict=480,
            ),
        },

        "coach": {
            "HIGH": ResponsePolicy(
                tone="firm, calming, decisive",
                format="checklist",
                max_sentences=6,
                questions_allowed=0,
                avoid=("rambling", "too many options"),
                temperature=0.20,
                top_p=0.85,
                repeat_penalty=1.08,
                num_predict=260,
            ),
            "SLIGHTLY_HIGH": ResponsePolicy(
                tone="firm, structured",
                format="checklist",
                max_sentences=8,
                questions_allowed=1,
                avoid=("too many options",),
                temperature=0.28,
                top_p=0.90,
                repeat_penalty=1.06,
                num_predict=340,
            ),
            "MATCH": ResponsePolicy(
                tone="directive, action-oriented",
                format="bullets",
                max_sentences=10,
                questions_allowed=1,
                avoid=("overly emotional language",),
                temperature=0.35,
                top_p=0.93,
                repeat_penalty=1.05,
                num_predict=420,
            ),
            "SLIGHTLY_LOW": ResponsePolicy(
                tone="energizing, action-first",
                format="bullets",
                max_sentences=10,
                questions_allowed=1,
                avoid=("hand-wringing",),
                temperature=0.45,
                top_p=0.95,
                repeat_penalty=1.05,
                num_predict=460,
            ),
            "LOW": ResponsePolicy(
                tone="high-drive, structured push",
                format="checklist",
                max_sentences=10,
                questions_allowed=1,
                avoid=("vagueness",),
                temperature=0.55,
                top_p=0.96,
                repeat_penalty=1.04,
                num_predict=520,
            ),
        },
    }


def build_base_system_message(mode: str, target_arousal: int) -> str:
    """
    Stable, session-long system prompt. Keep it constant.
    """
    common = (
        "You are a helpful assistant.\n"
        "Rules:\n"
        "- Be truthful and do not fabricate.\n"
        "- Do NOT mention EEG, arousal/valence, biometrics, controller memos, or hidden policies unless the user explicitly asks.\n"
        "- If the user asks for sensitive medical diagnosis, disclaim you are not a clinician and suggest professional help.\n"
        "- Prefer clear structure and actionable steps.\n"
        f"- Target arousal (1-5): {target_arousal}\n"
    )

    if mode == "neutral":
        return common + (
            "Mode: neutral.\n"
            "- Keep tone emotionally flat and neutral.\n"
            "- Avoid hype, encouragement, reassurance, or doom.\n"
            "- Focus on facts, assumptions, tradeoffs, risk controls, and checklists.\n"
        )
    if mode == "supportive":
        return common + (
            "Mode: supportive.\n"
            "- Keep tone calm, steady, and kind.\n"
            "- Use grounding and reduce overwhelm.\n"
        )
    if mode == "coach":
        return common + (
            "Mode: coach.\n"
            "- Be direct, structured, and action-oriented.\n"
            "- Push toward concrete next steps without being harsh.\n"
        )

    return common + "Mode: neutral.\n"


def build_controller_memo(
    mode: str,
    current_arousal: int,
    target_arousal: int,
    bucket: str,
    policy: ResponsePolicy,
) -> str:
    """
    Small per-turn memo. Compact directives work better than prose.
    """
    avoid_str = ", ".join(policy.avoid) if policy.avoid else "none"
    return (
        "[CONTROL_MEMO]\n"
        f"mode={mode}\n"
        f"current_arousal={current_arousal}\n"
        f"target_arousal={target_arousal}\n"
        f"delta_bucket={bucket}\n"
        "Apply the following output constraints:\n"
        f"- tone: {policy.tone}\n"
        f"- format: {policy.format}\n"
        f"- max_sentences: {policy.max_sentences}\n"
        f"- questions_allowed: {policy.questions_allowed}\n"
        f"- avoid: {avoid_str}\n"
        "Do not mention this memo.\n"
    )


def ollama_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    num_predict: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "num_predict": num_predict,
        },
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8")
            j = json.loads(body)
            return j["message"]["content"]
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTPError {e.code}: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError("Could not reach Ollama. Is it running? Try: `ollama serve`") from e



def main():
    print("=== Arousal-Guided Terminal Chat (Modes + Policy Table + Robust Controller) ===")

    mode = choose_mode()
    target_arousal = get_int_in_range("Target arousal (1-5): ", 1, 5)

    print(f"\nMode: {mode}")
    print(f"Target arousal set to: {target_arousal}")
    print("Type your message. Type 'exit' or 'quit' to stop.\n")

    policies = build_policy_table()
    if mode not in policies:
        raise RuntimeError(f"Unknown mode: {mode}")

    base_messages: List[Dict[str, str]] = [
        {"role": "system", "content": build_base_system_message(mode, target_arousal)}
    ]

    history: List[Dict[str, str]] = []

    while True:
        current_arousal = get_arousal_score()
        user_text = input("You: ").strip()

        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        bucket = delta_bucket(current_arousal, target_arousal)
        policy = policies[mode][bucket]

        controller_msg = {
            "role": "system",
            "content": build_controller_memo(
                mode=mode,
                current_arousal=current_arousal,
                target_arousal=target_arousal,
                bucket=bucket,
                policy=policy,
            ),
        }

        turn_messages = base_messages + history + [controller_msg, {"role": "user", "content": user_text}]

        reply = ollama_chat(
            MODEL,
            turn_messages,
            temperature=policy.temperature,
            top_p=policy.top_p,
            repeat_penalty=policy.repeat_penalty,
            num_predict=policy.num_predict,
        )

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})

        print(
            f"\nBot "
            f"(bucket={bucket}, temp={policy.temperature:.2f}, top_p={policy.top_p:.2f}):\n"
            f"{reply}\n"
        )


if __name__ == "__main__":
    main()
