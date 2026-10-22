# Native-speaker tasks (Hindi, Kannada, English)

Self-contained checklist for the user (Madhan). Supports `15` D18 (en + hi + kn live) and gives a
clean, untuned ASR/triage number. Nothing here needs code; send the results back and Claude wires them in.

**People:** 1 Hindi speaker + 1 Kannada speaker (ideally NOT the people who recorded the first 90
clips). English: you. **Time:** about 45–60 min per speaker. **Kit:** phone voice recorder, quiet room.

---

## 0. Consent (before anything)

Send each speaker this on WhatsApp and keep their "yes":

> "I'm recording a few short emergency-style sentences for my final-year project (pre-arrival
> emergency triage, CMRIT). Your voice will be used only for testing speech recognition in this
> project, kept private on my laptop, and not uploaded publicly. Is that OK?"

---

## Part A — Check the 6 questions (≈10 min, Hindi + Kannada only)

Show the speaker their language's column. Ask: **"Is this how a normal person in Bengaluru would say
it? Is anything wrong or too formal? Is the spelling right?"**

| ID | English | Hindi | Kannada |
|---|---|---|---|
| Q1 | Is the person awake and responding to you? | क्या व्यक्ति होश में है और जवाब दे रहा है? | ವ್ಯಕ್ತಿಗೆ ಪ್ರಜ್ಞೆ ಇದೆಯಾ ಮತ್ತು ನೀವು ಮಾತಾಡಿಸಿದಾಗ ಪ್ರತಿಕ್ರಿಯೆ ಕೊಡ್ತಿದ್ದಾರಾ? |
| Q2 | Is the person breathing normally? | क्या व्यक्ति सामान्य तरीके से साँस ले रहा है? | ವ್ಯಕ್ತಿ ಸಾಮಾನ್ಯವಾಗಿ ಉಸಿರಾಡ್ತಿದ್ದಾರಾ? |
| Q3 | Is there visible bleeding? If yes, is it heavy? | क्या खून बह रहा है? अगर हाँ, तो क्या बहुत ज़्यादा खून बह रहा है? | ರಕ್ತ ಬರ್ತಿದೆಯಾ? ಬಂದರೆ, ತುಂಬಾ ರಕ್ತ ಬರ್ತಿದೆಯಾ? |
| Q4 | Can the person speak clearly? | क्या वो ठीक से बोल पा रहे हैं? | ವ್ಯಕ್ತಿ ಸ್ಪಷ್ಟವಾಗಿ ಮಾತಾಡ್ತಿದ್ದಾರಾ? |
| Q5 | How many people are injured? | कितने लोग घायल हुए हैं? | ಎಷ್ಟು ಜನರಿಗೆ ಗಾಯವಾಗಿದೆ? |
| Q6 | Was the person hit by a fast vehicle, or did they fall from a height? | क्या व्यक्ति को किसी तेज़ गाड़ी ने टक्कर मारी है, या वो ऊँचाई से गिरे हैं? | ವ್ಯಕ್ತಿಗೆ ವೇಗವಾಗಿ ಬಂದ ಗಾಡಿ ಡಿಕ್ಕಿ ಹೊಡೀತಾ, ಅಥವಾ ಎತ್ತರದಿಂದ ಬಿದ್ರಾ? |

Rules:
- **Q2 must keep the meaning "breathing *normally*?"** (not "is there difficulty breathing?"). This is
  a safety fix: for an unconscious person, "no difficulty" could mean "not breathing at all".
- Keep each question one short sentence; it is shown on screen and spoken aloud.

**Send back:** the corrected lines as text (copy-paste), marked Q1–Q6 per language. "OK as is" is a
valid answer.

---

## Part C — Record 10 test sentences (≈20–30 min per language, en + hi + kn)

File with the scenarios: `data/asr_test/heldout/PROMPTS.md` (h01–h10). Answer sheet:
`data/asr_test/heldout/heldout_reference.csv`.

For each scenario h01 … h10:
1. The speaker **reads the FACTS column**, then **says it in their own natural words** — do **not** read
   the sample sentence word for word, and do not look at the app's keyword lists first.
2. One clip per scenario, **5–15 seconds**, normal speed, phone held normally (a couple in a noisy place
   is fine and realistic).
3. File names (same scenario number in every language):
   - English: `en_h01.wav` … `en_h10.wav` (or `.m4a` — then also change the extension in the CSV)
   - Hindi: `hi_h01.m4a` … `hi_h10.m4a`
   - Kannada: `kn_h01.m4a` … `kn_h10.m4a`
4. Put all clips in `data/asr_test/heldout/`.
5. Open `heldout_reference.csv` and, for each clip, **replace the `reference` text with exactly what was
   said** (in the speaker's own script: Devanagari / Kannada script; English loanwords written the way
   the speaker would write them). Keep the `file` column matching the real file names.

Tips:
- Listen to each clip once while typing the reference — it must be what was *actually* said, not the
  sample.
- If a speaker mixes English words (e.g. "accident", "bleeding"), that is good — write them as spoken.

**Send back:** "Part C done" — the files are already in the repo folder.

---

## Part B — Record the prompts (≈10 min per language, en + hi + kn; do this LAST)

Use the **final** wording from Part A. Read **clearly and slowly**, one file per line, 1 s of silence
at the start and end.

Questions: Q1–Q6 from the table above → `q1_<lang>.m4a` … `q6_<lang>.m4a`

Status lines (the speaker translates them first, then records):

| ID | English | Hindi | Kannada |
|---|---|---|---|
| S1 | Help is on the way. Stay with the person if it is safe. | मदद आ रही है। अगर सुरक्षित है, तो उस व्यक्ति के पास ही रहें। | ಸಹಾಯ ಬರುತ್ತಿದೆ. ನಿಮಗೆ ಸುರಕ್ಷಿತವಾಗಿದ್ದರೆ, ಆ ವ್ಯಕ್ತಿಯ ಜೊತೆಯಲ್ಲೇ ಇರಿ. |
| S2 | An ambulance has been sent. | एम्बुलेंस भेज दी गई है। | ಆಂಬ್ಯುಲೆನ್ಸ್ ಕಳಿಸಲಾಗಿದೆ. |
| S3 | Please answer a few quick questions. | बस कुछ छोटे-छोटे सवालों के जवाब दे दीजिए। | ದಯವಿಟ್ಟು ಕೆಲವು ಚಿಕ್ಕ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರ ಕೊಡಿ. |
| S4 | Follow these first-aid steps until help arrives. | मदद पहुँचने तक ये first-aid के तरीके अपनाइए। | ಸಹಾಯ ಬರುವವರೆಗೂ ಈ first-aid ಕ್ರಮಗಳನ್ನು ಅನುಸರಿಸಿ. |

Translations received 2026-10-22 and wired into `frontend/report.html` (S1 result, S2 after dispatch, S3 dialogue, S4 guidance).

→ `s1_<lang>.m4a` … `s4_<lang>.m4a`, where `<lang>` is `en`, `hi` or `kn`.

Put them in a new folder `data/prompts_audio/` (create it). Also send the **written** Hindi and Kannada
translations of S1–S4 as text.

---

## Order if short on time

1. Part A (hi + kn) — wrong wording would confuse real users.
2. Part C (en + hi + kn) — gives the clean result for the report / viva.
3. Part B — demo polish (browser speech is the fallback).

## What Claude does after you report back

- Updates `backend/dialogue/lexicon.py` with the Part A corrections (and re-runs the tests).
- Runs the held-out evaluation (`backend/asr/eval_triage.py`) on the Part C clips → clean hi/kn
  numbers, logged in `experiments/log.csv`, with Whisper and the new Kannada model (vasista22
  kannada-medium: 47.8 % WER / 15.4 % CER vs Whisper large-v3 80.1 % / 24.2 % on the 30 dev clips).
- Wires the Part B recordings into the report page (`frontend/report.html`).

**Do not commit the audio** if a speaker did not agree to it being stored in the repo; tell Claude and
the clips will stay local (gitignored).
