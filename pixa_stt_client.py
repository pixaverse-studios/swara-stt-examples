#!/usr/bin/env python3
"""
Luna STT Reference Client
=========================

A production-ready reference implementation for Luna Speech-to-Text API.
Supports both HTTP POST (pre-recorded audio) and WebSocket streaming (real-time).

API Documentation: https://docs.heypixa.ai/documentation/

Authentication:
    - X-Pixa-Key: Bearer <key>  (default)
    - Authorization: Bearer <key>
    - Authorization: Token <key>
    - Query parameter: ?api_key=<key>

Usage Examples:
    # HTTP POST (pre-recorded file)
    python pixa_stt_client.py listen --file audio.wav --api-key YOUR_KEY

    # WebSocket streaming (raw PCM file)
    python pixa_stt_client.py stream --file audio.pcm --api-key YOUR_KEY --dump-messages

    # Real-time streaming simulation
    python pixa_stt_client.py stream --file audio.pcm --api-key YOUR_KEY --realtime

Requirements:
    pip install requests websockets
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.parse
import wave
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

import requests
import websockets
from websockets.exceptions import InvalidStatus


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_HTTP_BASE_URL = "https://transcript.heypixa.ai"
DEFAULT_WS_BASE_URL = "wss://transcript.heypixa.ai"
LISTEN_PATH = "/v1/listen"

# Audio defaults
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_ENCODING = "linear16"
DEFAULT_LANGUAGE = "hi"


# =============================================================================
# Exceptions
# =============================================================================

class LunaClientError(RuntimeError):
    """Base exception for Luna client errors."""
    pass


class AuthenticationError(LunaClientError):
    """Raised when authentication fails."""
    pass


class ConnectionError(LunaClientError):
    """Raised when connection to server fails."""
    pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TranscriptResult:
    """Result from transcription."""
    transcript: str
    is_final: bool
    duration: float = 0.0
    start: float = 0.0
    request_id: str = ""


# =============================================================================
# Authentication Helpers
# =============================================================================

def _redact_key(key: str) -> str:
    """Redact API key for safe logging."""
    if not key or len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def load_api_key(explicit: Optional[str]) -> str:
    """
    Load API key from argument or environment variable.
    
    Args:
        explicit: Explicitly provided API key (takes precedence)
        
    Returns:
        The API key
        
    Raises:
        LunaClientError: If no API key is found
    """
    key = (explicit or os.environ.get("PIXA_API_KEY", "")).strip()
    if not key:
        raise LunaClientError(
            "Missing API key. Provide --api-key or set PIXA_API_KEY environment variable."
        )
    return key


def build_headers(api_key: str, auth_mode: str) -> Dict[str, str]:
    """
    Build authentication headers based on auth mode.
    
    Supported modes:
        - x-pixa-key-bearer: X-Pixa-Key: Bearer <key>
        - authorization-bearer: Authorization: Bearer <key>
        - authorization-token: Authorization: Token <key>
        - query-api-key: (no headers, key passed in URL)
    """
    headers = {
        "x-pixa-key-bearer": {"X-Pixa-Key": f"Bearer {api_key}"},
        "authorization-bearer": {"Authorization": f"Bearer {api_key}"},
        "authorization-token": {"Authorization": f"Token {api_key}"},
        "query-api-key": {},
    }
    if auth_mode not in headers:
        raise LunaClientError(f"Unknown auth mode: {auth_mode}")
    return headers[auth_mode]


def add_auth_query_param(url: str, api_key: str, auth_mode: str) -> str:
    """Add API key as query parameter if using query auth mode."""
    if auth_mode != "query-api-key":
        return url
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("api_key", api_key))
    new_query = urllib.parse.urlencode(query)
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, new_query, parsed.fragment)
    )


# =============================================================================
# Audio Helpers
# =============================================================================

def load_pcm_file(file_path: str) -> bytes:
    """Load raw PCM audio file."""
    with open(file_path, "rb") as f:
        return f.read()


def load_wav_file(file_path: str) -> Tuple[bytes, int, int]:
    """
    Load WAV file and extract PCM data.
    
    Returns:
        Tuple of (pcm_bytes, sample_rate, channels)
        
    Raises:
        LunaClientError: If file format is unsupported
    """
    try:
        with wave.open(file_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            pcm_frames = wf.readframes(wf.getnframes())
    except wave.Error as e:
        raise LunaClientError(f"Failed to read WAV file: {e}") from e

    if sample_width != 2:
        raise LunaClientError(
            f"WAV must be 16-bit PCM (linear16). Got {sample_width * 8}-bit."
        )
    
    return pcm_frames, sample_rate, channels


def iter_chunks(data: bytes, chunk_size: int) -> Iterator[bytes]:
    """Iterate over data in fixed-size chunks."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


# =============================================================================
# HTTP Client
# =============================================================================

def http_transcribe(
    *,
    base_url: str = DEFAULT_HTTP_BASE_URL,
    api_key: str,
    auth_mode: str = "x-pixa-key-bearer",
    file_path: str,
    language: str = DEFAULT_LANGUAGE,
    content_type: str = "audio/wav",
    timeout: float = 120.0,
    extra_params: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Transcribe audio using HTTP POST endpoint.
    
    Args:
        base_url: API base URL
        api_key: Your Luna API key
        auth_mode: Authentication mode
        file_path: Path to audio file
        language: Language code (e.g., 'hi' for Hindi)
        content_type: MIME type of audio file
        timeout: Request timeout in seconds
        extra_params: Additional query parameters
        
    Returns:
        Full API response as dictionary
        
    Raises:
        LunaClientError: On API error
    """
    # Build URL with query parameters
    url = f"{base_url}{LISTEN_PATH}"
    params = {"language": language}
    if extra_params:
        params.update(extra_params)
    url = f"{url}?{urllib.parse.urlencode(params)}"
    url = add_auth_query_param(url, api_key, auth_mode)

    # Build headers
    headers = build_headers(api_key, auth_mode)
    headers["Content-Type"] = content_type

    # Send request
    with open(file_path, "rb") as f:
        audio_data = f.read()

    response = requests.post(url, headers=headers, data=audio_data, timeout=timeout)

    try:
        payload = response.json()
    except Exception:
        payload = {"_raw": response.text}

    if response.status_code == 401:
        raise AuthenticationError(f"Invalid API key: {response.text}")
    if response.status_code >= 400:
        raise LunaClientError(f"HTTP {response.status_code}: {payload}")

    return payload


def extract_http_transcript(response: Dict[str, Any]) -> Optional[str]:
    """
    Extract transcript from HTTP response.
    
    Response format:
        {
            "results": {
                "channels": [{
                    "alternatives": [{
                        "transcript": "..."
                    }]
                }]
            }
        }
    """
    try:
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]
    except (KeyError, IndexError, TypeError):
        return None


# =============================================================================
# WebSocket Client
# =============================================================================

async def ws_transcribe(
    *,
    base_url: str = DEFAULT_WS_BASE_URL,
    api_key: str,
    auth_mode: str = "x-pixa-key-bearer",
    file_path: str,
    language: str = DEFAULT_LANGUAGE,
    encoding: str = DEFAULT_ENCODING,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk_ms: int = 100,
    realtime: bool = False,
    timeout: float = 180.0,
    extra_params: Optional[Dict[str, str]] = None,
    on_message: Optional[callable] = None,
) -> TranscriptResult:
    """
    Stream audio for transcription using WebSocket.
    
    Args:
        base_url: WebSocket base URL
        api_key: Your Luna API key
        auth_mode: Authentication mode
        file_path: Path to raw PCM file (linear16, mono)
        language: Language code
        encoding: Audio encoding (only 'linear16' supported)
        sample_rate: Audio sample rate in Hz
        chunk_ms: Chunk duration in milliseconds
        realtime: Simulate real-time streaming with delays
        timeout: Overall timeout in seconds
        extra_params: Additional query parameters
        on_message: Callback for each message (for streaming display)
        
    Returns:
        Final transcription result
        
    Raises:
        LunaClientError: On API error
    """
    if encoding.lower() != "linear16":
        raise LunaClientError(f"Only 'linear16' encoding is supported. Got: {encoding}")

    # Build WebSocket URL
    url = f"{base_url}{LISTEN_PATH}"
    params = {
        "language": language,
        "encoding": encoding,
        "sample_rate": str(sample_rate),
    }
    if extra_params:
        params.update(extra_params)
    url = f"{url}?{urllib.parse.urlencode(params)}"
    url = add_auth_query_param(url, api_key, auth_mode)

    # Build headers
    headers = build_headers(api_key, auth_mode)

    # Load audio
    pcm_data = load_pcm_file(file_path)
    bytes_per_sample = 2  # 16-bit PCM
    chunk_size = int(sample_rate * (chunk_ms / 1000.0) * bytes_per_sample)

    if chunk_size <= 0:
        raise LunaClientError("Invalid chunk size. Check sample_rate and chunk_ms.")

    # State
    final_result: Optional[TranscriptResult] = None
    accumulated_text = ""

    async def receive_messages(ws):
        """Receive and process server messages."""
        nonlocal final_result, accumulated_text
        
        async for msg in ws:
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                data = {"type": "Unknown", "raw": msg}

            if on_message:
                on_message(data)

            msg_type = data.get("type")

            if msg_type == "Metadata":
                # Session started
                continue

            elif msg_type == "Results":
                # Transcription result (streaming)
                # Response format: {"type": "Results", "transcript": "...", "is_final": bool, ...}
                transcript = data.get("transcript", "")
                is_final = data.get("is_final", True)
                
                if is_final:
                    # Accumulate final segments
                    if accumulated_text and transcript:
                        accumulated_text += " " + transcript
                    else:
                        accumulated_text = transcript
                        
                    final_result = TranscriptResult(
                        transcript=accumulated_text,
                        is_final=True,
                        duration=data.get("duration", 0.0),
                        start=data.get("start", 0.0),
                        request_id=data.get("metadata", {}).get("request_id", ""),
                    )

            elif msg_type == "Error":
                error_msg = data.get("message", "Unknown error")
                raise LunaClientError(f"Server error: {error_msg}")

    async def send_audio(ws):
        """Send audio chunks to server."""
        for chunk in iter_chunks(pcm_data, chunk_size):
            await ws.send(chunk)
            if realtime:
                await asyncio.sleep(chunk_ms / 1000.0)
        
        # Signal end of audio
        await ws.send(json.dumps({"type": "Finalize"}))

    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            # Run send and receive concurrently
            recv_task = asyncio.create_task(receive_messages(ws))
            send_task = asyncio.create_task(send_audio(ws))

            t0 = time.time()
            await send_task
            
            # Wait for final response with remaining timeout
            remaining = max(0.1, timeout - (time.time() - t0))
            try:
                await asyncio.wait_for(recv_task, timeout=remaining)
            except asyncio.TimeoutError:
                raise LunaClientError(f"Timeout waiting for transcription (timeout={timeout}s)")

            if final_result is None:
                raise LunaClientError("No transcription result received")

            return final_result

    except InvalidStatus as e:
        if "401" in str(e):
            raise AuthenticationError(f"Invalid API key: {e}") from e
        raise ConnectionError(f"WebSocket connection rejected: {e}") from e


# =============================================================================
# CLI
# =============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    
    # Common arguments parser (shared by all subcommands)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--api-key",
        help="API key (or set PIXA_API_KEY environment variable)",
    )
    common.add_argument(
        "--auth-mode",
        default="x-pixa-key-bearer",
        choices=["x-pixa-key-bearer", "authorization-bearer", "authorization-token", "query-api-key"],
        help="Authentication mode (default: x-pixa-key-bearer)",
    )
    common.add_argument(
        "--http-base-url",
        default=DEFAULT_HTTP_BASE_URL,
        help=f"HTTP base URL (default: {DEFAULT_HTTP_BASE_URL})",
    )
    common.add_argument(
        "--ws-base-url",
        default=DEFAULT_WS_BASE_URL,
        help=f"WebSocket base URL (default: {DEFAULT_WS_BASE_URL})",
    )

    # Main parser
    parser = argparse.ArgumentParser(
        prog="pixa_stt_client",
        description="Luna STT Reference Client - Speech-to-Text API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a WAV file
  %(prog)s listen --file audio.wav --api-key YOUR_KEY

  # Stream a raw PCM file
  %(prog)s stream --file audio.pcm --api-key YOUR_KEY --dump-messages

  # Real-time streaming simulation
  %(prog)s stream --file audio.pcm --api-key YOUR_KEY --realtime

Environment Variables:
  PIXA_API_KEY    Your Luna API key (alternative to --api-key)
        """,
        parents=[common],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------------------------------------------------------------------------
    # 'listen' subcommand (HTTP POST)
    # -------------------------------------------------------------------------
    listen_parser = subparsers.add_parser(
        "listen",
        help="Transcribe pre-recorded audio file (HTTP POST)",
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    listen_parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to audio file (WAV, MP3, etc.)",
    )
    listen_parser.add_argument(
        "--language", "-l",
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE})",
    )
    listen_parser.add_argument(
        "--content-type",
        default="audio/wav",
        help="Content-Type header (default: audio/wav)",
    )
    listen_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120)",
    )
    listen_parser.add_argument(
        "--dump-json",
        action="store_true",
        help="Print full JSON response",
    )

    # -------------------------------------------------------------------------
    # 'stream' subcommand (WebSocket)
    # -------------------------------------------------------------------------
    stream_parser = subparsers.add_parser(
        "stream",
        help="Stream audio for real-time transcription (WebSocket)",
        parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    stream_parser.add_argument(
        "--file", "-f",
        required=True,
        help="Path to raw PCM file (linear16, mono, 16kHz)",
    )
    stream_parser.add_argument(
        "--language", "-l",
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE})",
    )
    stream_parser.add_argument(
        "--sample-rate", "-r",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})",
    )
    stream_parser.add_argument(
        "--chunk-ms",
        type=int,
        default=100,
        help="Chunk duration in milliseconds (default: 100)",
    )
    stream_parser.add_argument(
        "--realtime",
        action="store_true",
        help="Simulate real-time streaming (add delays between chunks)",
    )
    stream_parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Overall timeout in seconds (default: 180)",
    )
    stream_parser.add_argument(
        "--dump-messages",
        action="store_true",
        help="Print all server messages",
    )

    args = parser.parse_args(argv)

    try:
        api_key = load_api_key(args.api_key)
        print(f"Using auth_mode={args.auth_mode}, api_key={_redact_key(api_key)}")

        # ---------------------------------------------------------------------
        # HTTP POST
        # ---------------------------------------------------------------------
        if args.command == "listen":
            print(f"Transcribing: {args.file}")
            t0 = time.time()
            
            response = http_transcribe(
                base_url=args.http_base_url,
                api_key=api_key,
                auth_mode=args.auth_mode,
                file_path=args.file,
                language=args.language,
                content_type=args.content_type,
                timeout=args.timeout,
            )
            
            elapsed = time.time() - t0
            transcript = extract_http_transcript(response)
            
            print(f"\n{'='*60}")
            print(f"Transcript: {transcript or '(none)'}")
            print(f"{'='*60}")
            print(f"Latency: {elapsed:.2f}s")
            
            if args.dump_json:
                print(f"\nFull Response:\n{json.dumps(response, indent=2, ensure_ascii=False)}")
            
            return 0

        # ---------------------------------------------------------------------
        # WebSocket Streaming
        # ---------------------------------------------------------------------
        if args.command == "stream":
            print(f"Streaming: {args.file}")
            print(f"Sample rate: {args.sample_rate}Hz, Chunk: {args.chunk_ms}ms")
            if args.realtime:
                print("Mode: Real-time simulation")
            print()

            # Message callback for --dump-messages
            def on_message(msg):
                if args.dump_messages:
                    print(json.dumps(msg, ensure_ascii=False))

            t0 = time.time()
            
            result = asyncio.run(
                ws_transcribe(
                    base_url=args.ws_base_url,
                    api_key=api_key,
                    auth_mode=args.auth_mode,
                    file_path=args.file,
                    language=args.language,
                    sample_rate=args.sample_rate,
                    chunk_ms=args.chunk_ms,
                    realtime=args.realtime,
                    timeout=args.timeout,
                    on_message=on_message,
                )
            )
            
            elapsed = time.time() - t0
            
            print(f"\n{'='*60}")
            print(f"Transcript: {result.transcript or '(none)'}")
            print(f"{'='*60}")
            print(f"Latency: {elapsed:.2f}s")
            
            return 0

    except AuthenticationError as e:
        print(f"Authentication Error: {e}", file=sys.stderr)
        return 1
    except LunaClientError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
