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
    # v3 (2026-10-27): common day-to-day emergencies, same book-page rule
    "asthma_attack": ("Asthma attack", 180, 183),
    "dental_avulsion": ("Dental avulsion", 206, 208),
    "mammal_bites": ("Mammal bites", 242, 243),
    "insect_stings": ("Insect bites or stings", 244, 249),
    "snake_bites": ("Snakebites", 254, 257),
    "poisoning": ("Poisoning", 258, 263),
    "allergic_reaction": ("Allergic reaction and anaphylaxis", 279, 280),   # 281+: repeat + education
    "diabetic_emergency": ("Diabetic emergency", 291, 294),
    "feeling_faint": ("Feeling faint", 299, 303),
    "fever": ("Fever", 304, 307),
    "abdominal_pain": ("Abdominal pain", 308, 312),
    "emergency_childbirth": ("Emergency childbirth", 313, 320),
    "hyperthermia": ("Hyperthermia", 332, 336),
    "dehydration": ("Dehydration", 337, 342),
    "hypothermia": ("Hypothermia", 343, 348),
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
    # [ASSUMPTION] v3 keywords written by hand on 2026-10-27, before the v3 eval queries were run
    "asthma_attack": "asthma attack, wheezing, tight chest, known asthmatic cannot breathe",
    "dental_avulsion": "tooth knocked out, broken tooth, tooth fell out after a hit to the mouth",
    "mammal_bites": "dog bite, animal bite, monkey bite, cat bite, human bite, bitten by a stray dog",
    "insect_stings": "bee sting, wasp sting, insect bite, scorpion sting, stung",
    "snake_bites": "snake bite, cobra, viper, bitten by a snake",
    "poisoning": "poisoning, swallowed something harmful, drank kerosene, bleach, pesticide, cleaning liquid, swallowed a battery",
    "allergic_reaction": "allergic reaction, anaphylaxis, swelling of face or lips or tongue, rash and difficulty breathing after food or a sting",
    "diabetic_emergency": "diabetic emergency, low blood sugar, diabetic person shaky sweaty confused",
    "feeling_faint": "feeling faint, dizzy, light-headed, about to faint, fainting",
    "fever": "fever, high temperature, feels very hot, child with fever",
    "abdominal_pain": "stomach pain, abdominal pain, belly ache, severe pain in the stomach",
    "emergency_childbirth": "emergency childbirth, woman in labour, baby is coming, delivering a baby, pregnant woman giving birth",
    "hyperthermia": "heat stroke, heat exhaustion, too hot, sun stroke, overheated in the sun, stopped sweating",
    "dehydration": "dehydration, very thirsty, not drinking water, vomiting and diarrhoea, dry mouth",
    "hypothermia": "hypothermia, very cold, shivering, cold and wet, body too cold",
}
KEEP = {"Key action", "Early recognition", "First aid steps", "NOTE"}
STOP = {"Introduction", "Guidelines", "Good practice points", "Chain of survival behaviours",
        "Prevent and prepare", "Access help", "Education considerations", "Special case",
        "Context considerations", "Learner considerations", "Facilitation tips",
        "Facilitation tools", "Scientific foundation", "Education review",
        "Learning connections", "Local adaptation"}
# Per-topic fixes for PDF layout quirks (v3, checked by hand 2026-10-27). keep/stop: extra section
# headings; resume: a heading that restarts keeping after a STOP, only once "First aid steps" was
# seen; drop: regex on finished items (removed); sub: (regex, replacement) on finished items.
TOPIC_RULES = {
    # recognition is headed "Recognise"; mild-reaction steps follow "Local adaptation" (a STOP);
    # the CAUTION list is epinephrine side effects (medication)
    "allergic_reaction": {"keep": {"Recognise"}, "stop": {"CAUTION"}, "resume": {"Mild allergic reaction"}},
    # insect-avoidance education bullets end up under CAUTION (education text, not first aid)
    "insect_stings": {"drop": r"^- (Keep away from flowering|Wear shoes|When outside, cover|Wipe off any food|"
                              r"Before moving objects|Shake out shoes|Do not touch or disturb bee|"
                              r"Remain calm if attacked|Run and find shelter)|^If this is not possible, wear long"},
    "feeling_faint": {"drop": r"^Figure \d|^Resuscitation\.", "sub": [(r"\s*See figure \d+ page \d+\.", "")]},
    # the sentence after "iii." is a ml/kg volume (dropped by the medication filter)
    "dehydration": {"sub": [(r"\s+iii\.$", "")]},
}
# sentences aimed at dispatch centres / professionals, not bystanders
NON_BYSTANDER_RE = re.compile(r"dispatch|professional responder|learner|facilitat", re.IGNORECASE)
HEADER_RE = re.compile(r"^(\d+\s+\|\s+International first aid|First aid\s+–.*\|\s+\d+\s*$)")
ITEM_RE = re.compile(r"^(•|>|\d+\.\s)")


def read_pages(pdf: Path) -> dict[int, str]:
    from pypdf import PdfReader
    r = PdfReader(str(pdf))
    return {i + 1: (p.extract_text() or "") for i, p in enumerate(r.pages)}


def extract(lines: list[str], rules: dict | None = None) -> list[list[str]]:
    """Return kept sections as lists of joined items."""
    rules = rules or {}
    keep_h, stop_h, resume_h = KEEP | rules.get("keep", set()), STOP | rules.get("stop", set()), rules.get("resume", set())
    lines = [ln.strip() for ln in lines if ln.strip() and not HEADER_RE.match(ln.strip())]
    sections, cur, keep, steps_seen = [], None, False, False
    for i, s in enumerate(lines):
        if s in keep_h:
            keep, cur = True, []
            steps_seen |= s == "First aid steps"
            sections.append(cur)
            continue
        if s in stop_h:
            keep = False
            continue
        if not keep and steps_seen and s in resume_h:
            keep, cur = True, [s + ":"]
            sections.append(cur)
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
        rules = TOPIC_RULES.get(stem, {})
        secs = extract(lines, rules)
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
                for pat, rep in rules.get("sub", []):
                    clean = re.sub(pat, rep, clean)
                if clean and not ("drop" in rules and re.search(rules["drop"], clean.strip())):
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
