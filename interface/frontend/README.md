# Swiss Multilingual Voice & Robotics Web UI

Zero-credential browser client for the Swiss Multilingual Voice & Robotics Assistant.

## Features
- **Real-time 16 kHz 16-bit Mono PCM Capture:** Uses Web Audio API downsampling over `ws(s)://<host>/ws/voice`.
- **Voice Activity Detection & Barge-In Interruption:** Visual RMS meter, partial transcripts, sentence-chunked WAV playback, and instant playback/generation cancellation.
- **Multilingual & Dialect Controls:** Supports `auto`, `gsw` (all 8 SwissDial dialects including `vs` Wallisertiitsch), `de-DE`, `de-CH`, `fr-FR`, and `fr-CH`.
- **Gemini Robotics ER 2 & Safety Console:** Camera image upload with 2D spatial detection overlay, mandatory human confirmation cards, and immediate Emergency Stop (`E-STOP`).

## Security
Contains **zero** cloud API keys or service-account credentials. Served directly by FastAPI (`SERVE_FRONTEND=true`) or any static web server.
