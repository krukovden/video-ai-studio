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
from dataclasses import dataclass
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

# Sampling, pinned. Left alone the service samples at its own default, near 1.0,
# and asking the same model about the same reel twice returns different scores
# for the same phrase — the edit moves, and nothing in the diff says why. This is
# the only provider in the pipeline that can be told not to do that; the CLI ones
# have no such control and are made repeatable by the replay cache instead.
# `seed` matters because temperature 0 is greedy decoding, not a guarantee: ties
# still have to be broken somehow.
GENERATION_CONFIG = {"temperature": 0.0, "top_p": 1.0, "seed": 7}

# Published rates: roughly 300 tokens per second of video at default media
# resolution, roughly 100 at low, and this endpoint bills at the low rate — a
# 6-second silent clip measured 378 video tokens, or 63 a second, which is one
# 66-token frame per second with no audio track to add its own 32. Used only to
# tell a creator what a run will cost before it runs.
TOKENS_PER_SECOND = {"default": 300, "low": 100}

# An uploaded video is not usable the instant the bytes land: the service
# transcodes it first, and referencing it too early fails.
POLL_SECONDS = 2.0

# Codes worth trying again: the service is busy or rate-limiting, not refusing.
# A 4xx says the request itself is wrong and will be wrong next time too.
RETRYABLE_CODES = frozenset({429, 500, 502, 503, 504})
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 4.0


@dataclass(frozen=True)
class Usage:
    """What one call actually cost, as the API reported it.

    There is no balance endpoint and no quota header on this API, so a reply's
    own accounting is the only honest measure of what a run spent. Kept split by
    modality because video is the expensive part and the whole point of the
    budget settings is to control it.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    video_tokens: int = 0
    text_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _read_usage(document: dict) -> Usage | None:
    """The reply's own token accounting, or None when it did not report any."""
    raw = document.get("usage")
    if not isinstance(raw, dict):
        return None
    by_modality = {
        str(entry.get("modality", "")).lower(): int(entry.get("tokens", 0))
        for entry in raw.get("input_tokens_by_modality") or []
        if isinstance(entry, dict)
    }
    return Usage(
        input_tokens=int(raw.get("total_input_tokens", 0)),
        output_tokens=int(raw.get("total_output_tokens", 0)),
        video_tokens=by_modality.get("video", 0),
        text_tokens=by_modality.get("text", 0),
    )


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
        # What the most recent call reported spending. The caller records it, so
        # a finished run can say what it cost rather than what it estimated.
        self.last_usage: Usage | None = None

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
        document = self._generate_document(body, timeout)
        self.last_usage = _read_usage(document)
        return _reply_text(document)

    def _generate_document(self, body: dict, timeout: int) -> dict:
        """Ask for a completion, retrying only what is worth retrying.

        A busy service answers 500 "high demand", and throwing away a reel that
        has just been uploaded — fifteen minutes of footage, already paid for in
        time — because of a temporary spike is the wrong response. A 4xx is not
        retried: the request is wrong and will be just as wrong next time.
        """
        payload = json.dumps(body).encode("utf-8")
        last: Exception | None = None
        for attempt in range(RETRY_ATTEMPTS):
            request = urllib.request.Request(
                INTERACTIONS_URL,
                data=payload,
                headers=self._headers({"Content-Type": "application/json"}),
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                failure = RuntimeError(f"Gemini API failed ({error.code}): {detail}")
                if error.code not in RETRYABLE_CODES:
                    raise failure from error
                last = failure
            except urllib.error.URLError as error:
                last = RuntimeError(f"Gemini API unreachable: {error}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (2 ** attempt))
        raise last if last else RuntimeError("Gemini API failed for an unknown reason")

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
        # one, so there is nothing being left on the table. The sampling controls
        # travel in the same generation_config it refused that one key from.
        body: dict = {
            "model": self.model,
            "input": parts,
            "generation_config": dict(GENERATION_CONFIG),
        }
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
