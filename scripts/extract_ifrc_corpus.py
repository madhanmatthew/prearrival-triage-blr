"""Extract bystander first-aid steps from the IFRC 2020 Guidelines PDF into the RAG corpus.

Source: IFRC, International first aid, resuscitation, and education guidelines 2020
(`data/raw/first_aid/EN_GFARC_GUIDELINES_2020.pdf`, not committed). Licence (PDF p.2):
non-commercial copies allowed, source acknowledged.

Per topic (book page = PDF page, from the PDF's table of contents) we keep only the sections a
bystander acts on: Key action, Early recognition, First aid steps, NOTE. Introductions,
guideline evidence, education and facilitation text are skipped. Text is copied verbatim
(line-wrap joined); medication/dosage lines are dropped (AGENTS rule 6) and counted.
Output: `data/first_aid_corpus/<topic>_ifrc.md`. Review every file by hand afterwards.

    python scripts/extract_ifrc_corpus.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.rag.pipeline import is_medication  # noqa: E402

PDF = Path("data/raw/first_aid/EN_GFARC_GUIDELINES_2020.pdf")
OUT = Path("data/first_aid_corpus")
URL = "https://www.ifrc.org/sites/default/files/2022-02/EN_GFARC_GUIDELINES_2020.pdf"
LICENSE = "(c) IFRC 2020; non-commercial copies allowed with source acknowledged (PDF p.2)"
ACCESSED = "2026-10-06"

# topic file stem -> (IFRC topic title, first page, last page)
TOPICS = {
    "recovery_position": ("Unresponsive and breathing normally", 128, 133),
    "cpr_adult": ("Unresponsive and abnormal breathing (adolescent and adult)", 134, 145),
    "cpr_child": ("Unresponsive and abnormal breathing (baby and child)", 147, 153),
    "cpr_defibrillator": ("Unresponsive and abnormal breathing when a defibrillator is available", 154, 163),
    "choking": ("Choking", 170, 174),
    "breathing_difficulty": ("Breathing difficulties", 175, 179),
    "severe_bleeding": ("Severe bleeding", 187, 194),
    "chest_abdomen_injury": ("Chest and abdomen injuries", 195, 198),
    "amputation": ("Amputation", 199, 201),
    "cuts_grazes": ("Cuts and grazes", 202, 205),
    "burns": ("Burns", 213, 220),
    "fractures": ("Fractures, sprains and strains", 223, 227),
    "spinal_injury": ("Spinal injury", 228, 232),
    "head_injury": ("Head injury", 233, 237),
    "chest_pain": ("Chest pain", 264, 270),
    "stroke": ("Stroke", 271, 278),
    "shock": ("Shock", 288, 290),
    "seizure": ("Seizure", 295, 298),
}
# Plain-language search keywords per topic (embedded with the chunks, never shown as guidance).
# [ASSUMPTION] written by hand on 2026-10-06 after the v1 test run; v2 uses a fresh test set.
KEYWORDS = {
    "recovery_position": "unconscious but breathing, unresponsive and breathing normally, fainted, passed out, recovery position",
    "cpr_adult": "CPR, cardiac arrest, not breathing, unresponsive adult, collapsed, no pulse, chest compressions",
    "cpr_child": "CPR for a child or baby, infant not breathing, unresponsive child, kid not breathing",
    "cpr_defibrillator": "AED, defibrillator, shock the heart, cardiac arrest, not breathing",
    "choking": "choking, something stuck in the throat, food stuck, airway blocked, cannot speak or cough",
    "breathing_difficulty": "difficulty breathing, shortness of breath, breathless, gasping, hyperventilation, panic breathing",
    "severe_bleeding": "severe bleeding, heavy blood loss, bleeding wound, object stuck in wound, glass or knife in wound, tourniquet",
    "chest_abdomen_injury": "chest injury, abdominal injury, stomach wound, organs coming out, rib injury, chest wound",
    "amputation": "amputation, finger cut off, hand or limb cut off, severed body part",
    "cuts_grazes": "small cut, scrape, graze, minor wound, abrasion, light bleeding",
    "burns": "burn, scald, hot oil, fire, hot water, hot metal, chemical burn",
    "fractures": "broken bone, fracture, sprain, strain, bent arm or leg, swollen ankle, cannot walk",
    "spinal_injury": "spinal injury, neck injury, back injury, cannot feel legs, fall from height, do not move the person",
    "head_injury": "head injury, hit head, head wound, concussion, confused or vomiting after a head blow",
    "chest_pain": "chest pain, heart attack, pain spreading to the arm or jaw, sweating with chest pain",
    "stroke": "stroke, face drooping, slurred speech, arm weakness, sudden numbness, sudden confusion",
    "shock": "shock, pale cold clammy skin, weak and dizzy after blood loss, rapid pulse",
    "seizure": "seizure, fit, convulsion, epilepsy, jerking or shaking uncontrollably",
}
KEEP = {"Key action", "Early recognition", "First aid steps", "NOTE"}
STOP = {"Introduction", "Guidelines", "Good practice points", "Chain of survival behaviours",
        "Prevent and prepare", "Access help", "Education considerations", "Special case",
        "Context considerations", "Learner considerations", "Facilitation tips",
        "Facilitation tools", "Scientific foundation", "Education review",
        "Learning connections", "Local adaptation"}
# sentences aimed at dispatch centres / professionals, not bystanders
NON_BYSTANDER_RE = re.compile(r"dispatch|professional responder|learner|facilitat", re.IGNORECASE)
HEADER_RE = re.compile(r"^(\d+\s+\|\s+International first aid|First aid\s+–.*\|\s+\d+\s*$)")
ITEM_RE = re.compile(r"^(•|>|\d+\.\s)")


def read_pages(pdf: Path) -> dict[int, str]:
    from pypdf import PdfReader
    r = PdfReader(str(pdf))
    return {i + 1: (p.extract_text() or "") for i, p in enumerate(r.pages)}


def extract(lines: list[str]) -> list[list[str]]:
    """Return kept sections as lists of joined items."""
    lines = [ln.strip() for ln in lines if ln.strip() and not HEADER_RE.match(ln.strip())]
    sections, cur, keep = [], None, False
    for i, s in enumerate(lines):
        if s in KEEP:
            keep, cur = True, []
            sections.append(cur)
            continue
        if s in STOP:
            keep = False
            continue
        if not keep:
            continue
        s = re.sub(r"\*+$", "", s).replace("Â", "").strip()
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        prev_closed = not cur or cur[-1].endswith((".", ":", "?", ")"))
        if (len(s.split()) <= 5 and s[:1].isupper() and not ITEM_RE.match(s)
                and not s.endswith((".", ",", "?", ":", ")")) and (prev_closed or ITEM_RE.match(nxt))):
            cur.append(s + ":")                      # subheading, e.g. "Severe choking"
            continue
        new_item = (not cur or ITEM_RE.match(s)
                    or (cur[-1].endswith((".", ":", "?")) and s[:1].isupper()))
        if new_item:
            cur.append(s)
        else:
            cur[-1] = f"{cur[-1]} {s}"
    return [sec for sec in sections if sec]


def clean_item(item: str) -> tuple[str, int]:
    """Drop medication (rule 6) and non-bystander sentences inside one item; keep its marker."""
    m = re.match(r"^(•|>|\d+\.)\s*", item)
    marker, text = (m.group(1), item[m.end():]) if m else ("", item)
    sents = re.split(r"(?<=[.!?:])\s+(?=[A-Z“\"(])|\s+(?=[a-h]\.\s)", text)  # + "a. b." sub-items
    keep = [x for x in sents if not is_medication(x) and not NON_BYSTANDER_RE.search(x)]
    dropped = sum(is_medication(x) for x in sents)
    if not keep:
        return "", dropped
    text = " ".join(keep)
    marker = {"•": "- ", ">": "  - "}.get(marker, f"{marker} " if marker else "")
    return marker + text, dropped


def drop_orphan_headings(items: list[str]) -> list[str]:
    """Remove trailing subheadings ("Mild choking:") with no content under them."""
    out = list(items)
    while out and _is_heading(out[-1]):
        out.pop()
    return out


def _is_heading(it: str) -> bool:
    return it.endswith(":") and len(it.split()) <= 5 and not it.startswith(("-", " "))


def main() -> None:
    pages = read_pages(PDF)
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, (title, a, b) in TOPICS.items():
        lines = [ln for p in range(a, b + 1) for ln in pages[p].splitlines()]
        secs = extract(lines)
        dropped, body, seen = 0, [], set()
        for sec in secs:
            # The PDF repeats blocks when a topic has adult + baby variants: drop the leading run
            # of already-emitted items (incl. subheadings), then any repeated non-heading item.
            j = 0
            while j < len(sec) and sec[j] in seen:
                j += 1
            sec = [it for it in sec[j:] if _is_heading(it) or it not in seen]
            seen.update(sec)
            kept = []
            for it in sec:
                clean, d = clean_item(it)
                dropped += d
                if clean:
                    kept.append(clean)
            kept = drop_orphan_headings(kept)
            if kept:
                body.append("\n".join(kept))
        header = (f"---\ntitle: International first aid, resuscitation, and education guidelines "
                  f"2020 - {title}\npublisher: IFRC\nurl: {URL}\nlicense: {LICENSE}\n"
                  f"topic: {stem}\nkeywords: {KEYWORDS[stem]}\npages: {a}-{b}\n"
                  f"accessed: {ACCESSED}\n---\n")
        (OUT / f"{stem}_ifrc.md").write_text(header + "\n\n".join(body) + "\n", encoding="utf-8")
        n = sum(len(x.splitlines()) for x in body)
        print(f"{stem:22s} pages {a}-{b}: {n:3d} lines, {dropped} medication sentences dropped")


if __name__ == "__main__":
    main()
