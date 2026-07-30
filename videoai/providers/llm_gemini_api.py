"""LLM through the metered Gemini API — the only provider that watches the video.

Every other provider in this pipeline is handed a transcript and a few stills.
That is enough to judge what was *said*, and structurally blind to how it was
*delivered*: comic timing, a reaction building on a face, a moment lasting
exactly as long as it should, the second a thing actually gives way. For a quiet
subject those are the whole edit, and none of them survive being turned into
text.

This provider uploads the clips themselves. Gemini samples video at 1 fps and
takes the audio track with it, so the model hears the take as well as seeing it.
That costs real money — see `video_token_estimate` — which is exactly why it is
opt-in per stage rather than the default.

That 1 fps matters for how far its timestamps can be trusted. Measured against a
live key: a sustained event at 12.0s in a 20-second clip came back as 12, while a
0.3-second flash in a 6-second clip was missed entirely and reported at the wrong
end of the file. So an observation from this provider is good to about a second,
and is treated downstream as evidence to be checked rather than a frame-accurate
cue.

Why the free CLI is not this: the Gemini CLI's individual OAuth tier was
withdrawn ("IneligibleTierError: this client is no longer supported for Gemini
Code Assist for individuals"), and even while it worked it had no way to pass a
video file to the model. An API key is the only route to video, and several
models carry a free tier of their own.

No SDK: the three calls this needs — start an upload, send the bytes, ask for a
completion — are plain HTTPS, and a dependency that pulls in its own transport
stack is a poor trade for that.
"""
from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from videoai.providers.json_reply import extract_json

SYSTEM_PROMPT = (
    "You are a video editing analyst. You reply with a single JSON document and "
    "nothing else: no prose, no markdown fences."
)

BASE = "https://generativelanguage.googleapis.com"
UPLOAD_URL = f"{BASE}/upload/v1beta/files"
INTERACTIONS_URL = f"{BASE}/v1beta/interactions"

# A flash-lite tier is the sane default: it reads video at a fraction of Pro's
# input price, and this pipeline asks it to score phrases and report what it saw
# rather than to reason hard. Verified against a live key — note that
# `gemini-2.5-flash` is refused for new accounts ("no longer available to new
# users"), so it is not a safe default even though it is still listed.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Published rates: roughly 300 tokens per second of video at default media
# resolution, roughly 100 at low, and this endpoint bills at the low rate — a
# 6-second silent clip measured 378 video tokens, or 63 a second, which is one
# 66-token frame per second with no audio track to add its own 32. Used only to
# tell a creator what a run will cost before it runs.
TOKENS_PER_SECOND = {"default": 300, "low": 100}

# An uploaded video is not usable the instant the bytes land: the service
# transcodes it first, and referencing it too early fails.
POLL_SECONDS = 2.0


def video_token_estimate(seconds: float, media_resolution: str = "default") -> int:
    """Roughly what submitting `seconds` of footage will cost in input tokens."""
    rate = TOKENS_PER_SECOND.get(media_resolution or "default", TOKENS_PER_SECOND["default"])
    return int(round(seconds * rate))


def _mime_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    return "video/mp4" if path.suffix.lower() in {".mp4", ".mov", ".m4v"} else "application/octet-stream"


def _part_type(mime: str) -> str:
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    return "document"


class GeminiApiLLM:
    name = "gemini_api"
    # The point of this provider.
    reads_video = True
    system_preamble = SYSTEM_PROMPT

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    # ---- the three HTTP seams, kept small so tests can replace them ---------

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"x-goog-api-key": self.api_key}
        headers.update(extra or {})
        return headers

    def _upload(self, path: Path, timeout: int) -> str:
        """Upload one file with the resumable protocol; return its `files/...` uri."""
        payload = path.read_bytes()
        mime = _mime_type(path)
        start = urllib.request.Request(
            UPLOAD_URL,
            data=json.dumps({"file": {"display_name": path.name}}).encode("utf-8"),
            headers=self._headers({
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(payload)),
                "X-Goog-Upload-Header-Content-Type": mime,
                "Content-Type": "application/json",
            }),
            method="POST",
        )
        with urllib.request.urlopen(start, timeout=timeout) as response:
            session_url = response.headers.get("x-goog-upload-url")
        if not session_url:
            raise RuntimeError(f"Gemini upload did not return a session url for {path.name}")

        send = urllib.request.Request(
            session_url,
            data=payload,
            headers=self._headers({
                "Content-Length": str(len(payload)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            }),
            method="POST",
        )
        with urllib.request.urlopen(send, timeout=timeout) as response:
            document = json.loads(response.read().decode("utf-8"))
        uri = (document.get("file") or {}).get("uri") or (document.get("file") or {}).get("name")
        if not uri:
            raise RuntimeError(f"Gemini upload returned no file uri for {path.name}")
        return uri

    def _await_active(self, uri: str, timeout: int) -> None:
        """Block until the service has finished transcoding an uploaded file."""
        name = uri.split("/files/")[-1] if "/files/" in uri else uri.removeprefix("files/")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            request = urllib.request.Request(
                f"{BASE}/v1beta/files/{name}", headers=self._headers(), method="GET"
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                state = json.loads(response.read().decode("utf-8")).get("state")
            if state == "ACTIVE":
                return
            if state == "FAILED":
                raise RuntimeError(f"Gemini could not process the uploaded file: {uri}")
            time.sleep(POLL_SECONDS)
        raise RuntimeError(f"Gemini did not finish processing {uri} within {timeout} seconds")

    def _generate(self, body: dict, timeout: int) -> str:
        request = urllib.request.Request(
            INTERACTIONS_URL,
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers({"Content-Type": "application/json"}),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                document = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Gemini API failed ({error.code}): {detail}") from error
        return _reply_text(document)

    # ---- the protocol ------------------------------------------------------

    def complete_json(
        self,
        prompt: str,
        images: list[Path],
        timeout: int,
        videos: list[Path] | None = None,
    ) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey "
                "and put it in .env — the Gemini CLI's free individual tier no longer works."
            )

        parts: list[dict] = [{"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{prompt}"}]
        for path in [*(videos or []), *images]:
            if not path.is_file():
                raise RuntimeError(f"cannot submit a file that does not exist: {path}")
            mime = _mime_type(path)
            uri = self._upload(path, timeout)
            self._await_active(uri, timeout)
            parts.append({"type": _part_type(mime), "uri": uri, "mime_type": mime})

        # No media_resolution: the interactions endpoint rejects it in every
        # position the documentation suggests (generation_config, top level, and
        # on the input part), and sending it fails the whole call. Measured
        # against a live key, video already tokenises at roughly 63 tokens per
        # second here, which is the low-resolution rate rather than the default
        # one, so there is nothing being left on the table.
        body: dict = {"model": self.model, "input": parts}
        return extract_json(self._generate(body, timeout))


def _reply_text(document: dict) -> str:
    """The model's own text, out of whichever response shape came back.

    The interactions API answers with a list of `steps`: the model's private
    reasoning is one step and its actual answer another, so the text has to be
    taken from the `model_output` step specifically — concatenating everything
    would feed the JSON reader the model's thinking as well.

    The older generateContent shapes are read too, so pointing this provider at
    a different endpoint later is not also a rewrite of its parsing.
    """
    steps = document.get("steps")
    if isinstance(steps, list):
        chunks: list[str] = []
        for step in steps:
            if not isinstance(step, dict) or step.get("type") != "model_output":
                continue
            for part in step.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
        if chunks:
            return "\n".join(chunks)

    for key in ("output_text", "text", "response"):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            return value

    output = document.get("output")
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                return item["text"]

    candidates = document.get("candidates")
    if isinstance(candidates, list) and candidates:
        content = candidates[0].get("content") or {}
        for part in content.get("parts") or []:
            if isinstance(part.get("text"), str):
                return part["text"]

    raise RuntimeError(f"Gemini API returned no text: {json.dumps(document)[:300]}")
