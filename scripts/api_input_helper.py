"""
End-to-end Multilingual Spoken Language Understanding (SLU) via Audio LLM API.

Maps raw audio directly to structured English commands:
    <intent> | <annotated English instruction>

No ASR / MT / NLU cascade — single API call.
"""

from __future__ import annotations

from typing import Any, Dict, List

import base64
from pathlib import Path

# 1. Prompt Configuration

SYSTEM_PROMPT = """\
You are an end-to-end multilingual spoken language understanding system.

# Task
Listen to the input audio and produce a single-line structured English command.

# Output Format
<intent> | <annotated English instruction>

# Rules
- Identify the speaker's intent (e.g. transport_ticket, calendar_remove, alarm_set, audio_volume_mute, weather_query, etc.).
- Translate the spoken content into natural English regardless of source language.
- Annotate every slot value inline: [slot_type : value]
- PRESERVE named entities (person names, place names) in the original spoken language. Do NOT translate proper nouns.
- Separate intent and instruction with " | ".
- Output ONLY the structured command. No explanation, no transcription, no extra text.

# Examples
Input: 🔊 (German audio — a speaker asking about train travel)
Output: transport_ticket | i'd like to get on the next [transport_type : train] that will be departing [place_name : münchen] headed to [place_name : köln]

Input: 🔊 (Spanish audio — a speaker asking to delete a calendar event)
Output: calendar_remove | please delete the [event_name : meeting] i have with [person : óscar] [timeofday : tonight]"""


USER_PROMPT = "Listen to the audio in {locale}  and output the structured command."


# Audio helpers


def encode_audio_base64(audio_path: str | Path) -> str:
    """Read an audio file and return its base64 encoding."""
    return base64.b64encode(Path(audio_path).read_bytes()).decode("utf-8")


def get_mime_type(audio_path: str | Path) -> str:
    """Infer MIME type from file extension."""
    suffix = Path(audio_path).suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mp3",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/m4a",
        ".webm": "audio/webm",
    }.get(suffix, "audio/wav")


# API request


def build_openai_request(audio_path: str | Path, locale: str) -> List[Dict[str, Any]]:
    """Build request body for OpenAI-compatible audio API."""
    audio_b64 = encode_audio_base64(audio_path)
    mime = get_mime_type(audio_path)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": mime.split("/")[-1],
                    },
                },
                {"type": "text", "text": USER_PROMPT.format(locale=locale)},
            ],
        },
    ]
