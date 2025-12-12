# Real-Time Streaming API Reference

> **Endpoint:** `wss://transcript.heypixa.ai/v1/listen`
>
> Stream audio and receive transcriptions in real-time via WebSocket.

---

## Overview

The Luna Streaming API provides real-time speech-to-text transcription with:

- **Streaming Results**: Partial transcripts as you speak
- **Low Latency**: Sub-second response times
- **High Accuracy**: Optimized for Indian dialects

Results are streamed incrementally as you speak—not just after the audio stream ends.

---

## Connection

### WebSocket URL

```
wss://transcript.heypixa.ai/v1/listen
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language` | string | `hi` | Language code for transcription |
| `encoding` | string | `linear16` | Audio encoding format |
| `sample_rate` | integer | `16000` | Sample rate in Hz |
| `api_key` | string | — | API key (if using query auth) |

### Example Connection URL

```
wss://transcript.heypixa.ai/v1/listen?language=hi&encoding=linear16&sample_rate=16000
```

---

## Authentication

WebSocket connections support multiple authentication methods:

### Header-Based (Recommended)

```javascript
// X-Pixa-Key header
const ws = new WebSocket(url, {
  headers: { 'X-Pixa-Key': 'Bearer YOUR_API_KEY' }
});

// Authorization header
const ws = new WebSocket(url, {
  headers: { 'Authorization': 'Bearer YOUR_API_KEY' }
});
```

### Query Parameter (Browser Fallback)

For browser environments where custom headers aren't supported:

```
wss://transcript.heypixa.ai/v1/listen?api_key=YOUR_API_KEY&language=hi
```

---

## Audio Format

### Supported Encodings

| Encoding | Description |
|----------|-------------|
| `linear16` | 16-bit signed PCM, little-endian **(recommended)** |
| `linear32` | 32-bit signed PCM, little-endian |
| `mulaw` | G.711 μ-law |
| `alaw` | G.711 A-law |

### Recommended Settings

| Setting | Value |
|---------|-------|
| Encoding | `linear16` |
| Sample Rate | `16000` Hz |
| Channels | Mono (1 channel) |
| Bit Depth | 16-bit |
| Byte Order | Little-endian |

### Chunk Size

Send audio in **100ms chunks** for optimal performance:

```
Chunk size = sample_rate × (chunk_ms / 1000) × bytes_per_sample
           = 16000 × 0.1 × 2
           = 3200 bytes
```

---

## Server Messages

### Metadata

Sent immediately when the connection is established:

```json
{
  "type": "Metadata",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "created": "2024-01-15T10:30:00Z",
  "channels": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"Metadata"` |
| `request_id` | string | Unique session identifier (UUID) |
| `created` | string | ISO 8601 timestamp |
| `channels` | integer | Number of audio channels |

### Results

Transcription results are streamed in real-time as speech is detected:

```json
{
  "type": "Results",
  "transcript": "नमस्ते आप कैसे हैं",
  "is_final": true,
  "duration": 2.5,
  "start": 0.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"Results"` |
| `transcript` | string | The transcribed text |
| `is_final` | boolean | `true` for final results, `false` for partials |
| `duration` | number | Audio duration in seconds |
| `start` | number | Start time offset in seconds |

#### Partial vs Final Results

- **Partial (`is_final: false`)**: Intermediate results that may change as more audio is processed. Use for real-time display.
- **Final (`is_final: true`)**: Confirmed transcription for a speech segment. Safe to commit to your application state.

### Error

Sent when an error occurs:

```json
{
  "type": "Error",
  "message": "Invalid audio format"
}
```

---

## Client Commands

### Send Audio

Send raw audio bytes (binary data) directly—no JSON wrapper:

```javascript
ws.send(audioChunk);  // ArrayBuffer or Uint8Array
```

### Finalize

Signal the end of the audio stream to flush any buffered audio:

```json
{"type": "Finalize"}
```

### CloseStream

Gracefully end the session:

```json
{"type": "CloseStream"}
```

### KeepAlive

Prevent connection timeout during silence:

```json
{"type": "KeepAlive"}
```

---

## Real-Time Streaming Flow

```
Client                                         Server
  │                                               │
  │  ──── WebSocket Connect ─────────────────►    │
  │  ◄─── Metadata {"request_id": "..."}  ────    │
  │                                               │
  │  ──── [audio chunk 1] ───────────────────►    │
  │  ──── [audio chunk 2] ───────────────────►    │
  │  ──── [audio chunk 3] ───────────────────►    │
  │  ◄─── Results {"is_final": false} ────────    │  ← Partial result
  │  ──── [audio chunk 4] ───────────────────►    │
  │  ──── [audio chunk 5] ───────────────────►    │
  │  ◄─── Results {"is_final": true} ─────────    │  ← Final result (segment complete)
  │  ──── [audio chunk 6] ───────────────────►    │
  │  ◄─── Results {"is_final": false} ────────    │  ← New segment partial
  │  ──── [audio chunk 7] ───────────────────►    │
  │  ──── {"type": "Finalize"} ──────────────►    │
  │  ◄─── Results {"is_final": true} ─────────    │  ← Final flush
  │  ◄─── Connection Close ───────────────────    │
  │                                               │
```

---

## Complete Examples

### JavaScript (Browser)

```javascript
class LunaStreamingClient {
  constructor(apiKey, language = 'hi') {
    this.apiKey = apiKey;
    this.language = language;
    this.ws = null;
  }

  connect() {
    const url = new URL('wss://transcript.heypixa.ai/v1/listen');
    url.searchParams.set('language', this.language);
    url.searchParams.set('encoding', 'linear16');
    url.searchParams.set('sample_rate', '16000');
    url.searchParams.set('api_key', this.apiKey);

    this.ws = new WebSocket(url.toString());

    this.ws.onopen = () => {
      console.log('Connected');
    };

    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      
      if (msg.type === 'Metadata') {
        console.log('Session:', msg.request_id);
      }
      
      if (msg.type === 'Results') {
        if (msg.is_final) {
          console.log('Final:', msg.transcript);
        } else {
          console.log('Partial:', msg.transcript);
        }
      }
      
      if (msg.type === 'Error') {
        console.error('Error:', msg.message);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    this.ws.onclose = (event) => {
      console.log('Disconnected:', event.code);
    };
  }

  sendAudio(pcmData) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(pcmData);
    }
  }

  finalize() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'Finalize' }));
    }
  }

  close() {
    this.finalize();
    setTimeout(() => this.ws?.close(), 500);
  }
}

// Usage
const client = new LunaStreamingClient('pk_your_api_key', 'hi');
client.connect();

// Stream audio from microphone
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    const audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(1024, 1, 1);
    
    processor.onaudioprocess = (e) => {
      const float32 = e.inputBuffer.getChannelData(0);
      const pcm16 = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        pcm16[i] = Math.max(-1, Math.min(1, float32[i])) * 0x7FFF;
      }
      client.sendAudio(pcm16.buffer);
    };
    
    source.connect(processor);
    processor.connect(audioContext.destination);
  });
```

### Python

```python
import asyncio
import json
import websockets

async def transcribe_stream(api_key: str, audio_file: str, language: str = 'hi'):
    url = (
        f"wss://transcript.heypixa.ai/v1/listen"
        f"?language={language}&encoding=linear16&sample_rate=16000"
    )
    headers = {"X-Pixa-Key": f"Bearer {api_key}"}
    
    async with websockets.connect(url, additional_headers=headers) as ws:
        # Handle messages
        async def receive():
            async for msg in ws:
                data = json.loads(msg)
                
                if data['type'] == 'Metadata':
                    print(f"Session: {data['request_id']}")
                
                elif data['type'] == 'Results':
                    prefix = "Final" if data['is_final'] else "Partial"
                    print(f"{prefix}: {data['transcript']}")
                
                elif data['type'] == 'Error':
                    print(f"Error: {data['message']}")
        
        # Send audio
        async def send():
            with open(audio_file, 'rb') as f:
                chunk_size = 3200  # 100ms at 16kHz, 16-bit
                while chunk := f.read(chunk_size):
                    await ws.send(chunk)
                    await asyncio.sleep(0.1)  # Real-time pacing
            
            await ws.send(json.dumps({"type": "Finalize"}))
        
        # Run concurrently
        await asyncio.gather(receive(), send())

# Usage
asyncio.run(transcribe_stream('pk_your_api_key', 'audio.pcm'))
```

### Node.js

```javascript
const WebSocket = require('ws');
const fs = require('fs');

const apiKey = 'pk_your_api_key';
const url = 'wss://transcript.heypixa.ai/v1/listen?language=hi&encoding=linear16&sample_rate=16000';

const ws = new WebSocket(url, {
  headers: { 'X-Pixa-Key': `Bearer ${apiKey}` }
});

ws.on('open', () => {
  console.log('Connected');
  
  // Stream audio file in chunks
  const audioStream = fs.createReadStream('audio.pcm', { highWaterMark: 3200 });
  
  audioStream.on('data', (chunk) => {
    ws.send(chunk);
  });
  
  audioStream.on('end', () => {
    ws.send(JSON.stringify({ type: 'Finalize' }));
  });
});

ws.on('message', (data) => {
  const msg = JSON.parse(data);
  
  if (msg.type === 'Metadata') {
    console.log('Session:', msg.request_id);
  }
  
  if (msg.type === 'Results') {
    const prefix = msg.is_final ? '✓' : '...';
    console.log(`${prefix} ${msg.transcript}`);
  }
});

ws.on('error', console.error);
ws.on('close', () => console.log('Disconnected'));
```

---

## Best Practices

### 1. Handle Partial Results

Display partial results for responsive UX, but only commit final results to your data:

```javascript
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'Results') {
    if (msg.is_final) {
      // Commit to database/state
      saveTranscript(msg.transcript);
    } else {
      // Display only (will be replaced)
      showPartialText(msg.transcript);
    }
  }
};
```

### 2. Implement Reconnection

Add exponential backoff for production reliability:

```javascript
function connectWithRetry(attempt = 1) {
  const ws = new WebSocket(url);
  
  ws.onclose = (event) => {
    if (event.code !== 1000) {
      const delay = Math.min(1000 * Math.pow(2, attempt), 30000);
      console.log(`Reconnecting in ${delay}ms...`);
      setTimeout(() => connectWithRetry(attempt + 1), delay);
    }
  };
  
  ws.onopen = () => {
    attempt = 1;  // Reset on successful connect
  };
  
  return ws;
}
```

### 3. Use KeepAlive for Long Sessions

Prevent timeout during extended silence:

```javascript
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'KeepAlive' }));
  }
}, 30000);  // Every 30 seconds
```

### 4. Always Finalize

Send `Finalize` before closing to ensure all audio is processed:

```javascript
function gracefulClose() {
  ws.send(JSON.stringify({ type: 'Finalize' }));
  setTimeout(() => ws.close(1000), 500);
}
```

---

## Error Handling

### Connection Errors

| Code | Description | Solution |
|------|-------------|----------|
| 401 | Invalid API key | Verify your API key |
| 403 | Forbidden | Check API key permissions |
| 429 | Rate limited | Implement backoff, reduce request rate |
| 1006 | Abnormal closure | Network issue, reconnect |

### Audio Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "Invalid audio format" | Wrong encoding | Use `linear16` at 16kHz |
| "Sample rate mismatch" | Incorrect `sample_rate` param | Match param to actual audio |
| No transcription | Silent audio or wrong format | Verify audio contains speech |

---

## Rate Limits

- **Concurrent connections**: Based on your plan
- **Audio duration**: No hard limit (streaming)
- **Requests per minute**: Plan-dependent

See [Rate Limits](/documentation/rate-limits/) for details.

---

## Related

- [Authentication](/documentation/authentication/) — API key setup
- [Pre-recorded Transcription](/documentation/speech-to-text/) — HTTP API for files
- [Error Handling](/documentation/error-handling/) — Error codes reference

