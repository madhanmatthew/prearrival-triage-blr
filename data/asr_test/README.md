# ASR test set (Whisper WER, docs/01 §8, docs/08 §1)

Team-recorded spoken emergency reports. Audio (`*.wav`) is gitignored: share it via Drive.
Commit only `reference.csv`.

- >= 30 utterances per language (`en`, `hi`, `kn`), 5-15 s each, real voices (not TTS), ideally
  several speakers, some with background street noise. Get consent from every speaker.
- Mono 16 kHz WAV preferred (any format ffmpeg/PyAV can read works).
- `reference.csv` (UTF-8): `file,language,reference`
  - `file`: WAV name relative to this folder, e.g. `kn_001.wav`
  - `language`: `en` | `hi` | `kn`
  - `reference`: exact words spoken, in the native script (Devanagari for Hindi, Kannada script
    for Kannada). Write numbers the way you say them, consistently.

Run: `make eval-asr` (or `python -m backend.asr.eval_wer`). First run downloads the Whisper model
(size from `WHISPER_MODEL`) into `models/whisper/`.

## Noise robustness (`make eval-asr-noise`)

Put 2-5 min of real Bengaluru street noise (traffic, crowd, wind; phone held at arm's length)
as `*.wav` in `data/asr_test/noise/` (gitignored, share via Drive). The eval mixes it into the
clean clips at 20/10/5/0 dB SNR (seeded) and reports WER per language x SNR, VAD on vs off, to
`reports/asr_wer_snr.csv`. Also record ~10 clips per language in real noise and add them to
`reference.csv` as normal rows. Mixed clips are synthetic noise: label them so. With no noise
wav the eval falls back to generated pink noise (weak stand-in). Nothing is trained.
Optional: `pip install noisereduce`, then `make eval-asr-noise ARGS=--denoise`.
