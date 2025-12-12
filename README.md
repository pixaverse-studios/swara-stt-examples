# Luna STT Reference Clients

Production-ready reference implementations for the Luna Speech-to-Text API.

## Overview

This directory contains reference client implementations demonstrating how to integrate with Luna STT:

| File | Description |
|------|-------------|
| `pixa_stt_client.py` | Python CLI client for HTTP and WebSocket APIs |
| `live_transcription_client.html` | Browser-based real-time transcription client |

## API Documentation

Full API documentation: https://docs.heypixa.ai/documentation/

### Endpoints

| Endpoint | Protocol | Description |
|----------|----------|-------------|
| `https://transcript.heypixa.ai/v1/listen` | HTTP POST | Pre-recorded audio transcription |
| `wss://transcript.heypixa.ai/v1/listen` | WebSocket | Real-time streaming transcription |

### Authentication

Luna API supports multiple authentication methods:

```bash
# Header: X-Pixa-Key (recommended)
X-Pixa-Key: Bearer YOUR_API_KEY

# Header: Authorization Bearer
Authorization: Bearer YOUR_API_KEY

# Header: Authorization Token
Authorization: Token YOUR_API_KEY

# Query Parameter
?api_key=YOUR_API_KEY
```

### Audio Format

| Parameter | Value | Description |
|-----------|-------|-------------|
| `encoding` | `linear16` | 16-bit signed PCM (little-endian) |
| `sample_rate` | `16000` | 16 kHz (recommended) |
| `channels` | `1` | Mono audio |

---

## Python Client

### Installation

```bash
cd test-stt
pip install -r requirements.txt
```

### HTTP Transcription

Transcribe pre-recorded audio files:

```bash
# Basic usage
python pixa_stt_client.py listen --file audio.wav --api-key YOUR_KEY

# With options
python pixa_stt_client.py listen \
    --file audio.wav \
    --api-key YOUR_KEY \
    --language hi \
    --dump-json
```

### WebSocket Streaming

Stream audio for real-time transcription:

```bash
# Stream a raw PCM file
python pixa_stt_client.py stream \
    --file audio.pcm \
    --api-key YOUR_KEY \
    --dump-messages

# Real-time simulation (adds delays between chunks)
python pixa_stt_client.py stream \
    --file audio.pcm \
    --api-key YOUR_KEY \
    --realtime
```

### Environment Variables

```bash
# Set API key via environment variable
export PIXA_API_KEY=pk_your_api_key

# Then use without --api-key flag
python pixa_stt_client.py stream --file audio.pcm
```

### Python Integration Example

```python
import asyncio
from pixa_stt_client import ws_transcribe, http_transcribe

# HTTP transcription
response = http_transcribe(
    api_key="YOUR_KEY",
    file_path="audio.wav",
    language="hi"
)
print(response)

# WebSocket streaming
async def stream_example():
    result = await ws_transcribe(
        api_key="YOUR_KEY",
        file_path="audio.pcm",
        language="hi",
        on_message=lambda msg: print(f"Received: {msg}")
    )
    print(f"Final: {result.transcript}")

asyncio.run(stream_example())
```

---

## Browser Client

### Usage

1. Open `live_transcription_client.html` in a modern browser
2. Enter your API key
3. Click "Start" to begin real-time transcription
4. Speak into your microphone
5. Click "Stop" when finished

### Features

- Real-time microphone capture via WebAudio API
- Automatic resampling (browser rate → 16kHz)
- Linear interpolation for high-quality downsampling
- Partial results displayed in purple, final in white
- Debug log for troubleshooting
- API key persistence (localStorage)

### Browser Requirements

- Modern browser with WebSocket support
- `getUserMedia` API (microphone access)
- WebAudio API (AudioContext)

Tested on: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+

---

## WebSocket Protocol

### Connection

```javascript
const ws = new WebSocket('wss://transcript.heypixa.ai/v1/listen?language=hi&encoding=linear16&sample_rate=16000&api_key=YOUR_KEY');
```

### Message Types

#### Server → Client

**Metadata** (sent on connection):
```json
{
    "type": "Metadata",
    "request_id": "uuid",
    "created": "2025-01-01T00:00:00Z"
}
```

**Results** (transcription output):
```json
{
    "type": "Results",
    "transcript": "transcribed text",
    "is_final": false,
    "duration": 1.5,
    "start": 0.0
}
```

- `is_final: false` — Partial result (may change)
- `is_final: true` — Final result (complete segment)

**Error**:
```json
{
    "type": "Error",
    "message": "error description"
}
```

#### Client → Server

**Audio data**: Send raw PCM bytes (no JSON wrapper)

**Finalize** (end session):
```json
{
    "type": "Finalize"
}
```

### Example Flow

```
Client                                    Server
   |                                         |
   |  --- WebSocket Connect -------------→   |
   |  ←-- Metadata {"request_id": ...}  ---  |
   |                                         |
   |  --- [raw PCM bytes] ---------------→   |
   |  --- [raw PCM bytes] ---------------→   |
   |  ←-- Results {"is_final": false} ----   |
   |  --- [raw PCM bytes] ---------------→   |
   |  ←-- Results {"is_final": true} -----   |
   |  --- {"type": "Finalize"} ----------→   |
   |  ←-- Connection Close ---------------   |
   |                                         |
```

---

## Audio Preparation

### Converting WAV to PCM

```bash
# Using ffmpeg
ffmpeg -i input.wav -f s16le -acodec pcm_s16le -ar 16000 -ac 1 output.pcm

# Using sox
sox input.wav -r 16000 -c 1 -b 16 -e signed-integer -t raw output.pcm
```

### Recording Test Audio

```bash
# Record 5 seconds of audio (Linux)
arecord -f S16_LE -r 16000 -c 1 -d 5 test.pcm

# Record using sox
sox -d -r 16000 -c 1 -b 16 test.pcm trim 0 5
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `401 Unauthorized` | Check API key is valid and properly formatted |
| No transcription output | Verify audio is 16-bit PCM at 16kHz, mono |
| High latency | Reduce chunk size, check network connection |
| Garbled output | Ensure sample rate matches the `sample_rate` parameter |

### Debug Mode

Python client:
```bash
python pixa_stt_client.py stream --file audio.pcm --api-key KEY --dump-messages
```

Browser client:
- Open the Debug Log panel
- Check browser console (F12)

---

## License

Copyright © 2025 Pixa. All rights reserved.
