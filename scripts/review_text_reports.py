"""Fast human review of the text-severity drafts (docs/08 §4, D2): one report at a time in the browser.

    python -m scripts.review_text_reports            # then open http://127.0.0.1:8770
    python -m scripts.review_text_reports --language hi   # only Hindi rows (for a native speaker)

Keys: Enter = OK (label correct, text natural) · 0/1/2/3 = set label and mark reviewed · E = edit text
· S = skip. Every action writes data/text_reports/reports.csv immediately (UTF-8 with BOM, same
columns), sets reviewed=1 and appends the reviewer's name to `notes`. Only reviewed rows are used
for training. Local tool, not part of the deployed app.
"""
from __future__ import annotations

import argparse
import threading
from pathlib import Path

import pandas as pd

CSV = Path("data/text_reports/reports.csv")
RUBRIC = {0: "stable — no injury / low concern", 1: "moderate — minor injury, stable",
          2: "severe — conscious but serious injury / controllable heavy bleeding",
          3: "critical — unconscious / not breathing / heavy uncontrolled bleeding / major trauma"}
_lock = threading.Lock()


def load(path: Path = CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"notes": str, "split": str})
    df["notes"] = df["notes"].fillna("")
    return df


def save(df: pd.DataFrame, path: Path = CSV) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def apply_review(df: pd.DataFrame, report_id: str, label: int | None, text: str | None, reviewer: str) -> pd.DataFrame:
    """Mark one row reviewed; optionally change its label and/or text. Returns the updated frame."""
    idx = df.index[df.report_id == report_id]
    if len(idx) != 1:
        raise KeyError(report_id)
    i = idx[0]
    if label is not None:
        if label not in RUBRIC:
            raise ValueError(f"label must be 0-3, got {label}")
        if int(df.at[i, "label"]) != label:
            df.at[i, "notes"] = (df.at[i, "notes"] + f" relabel {int(df.at[i, 'label'])}->{label};").strip()
        df.at[i, "label"] = label
    if text is not None and text.strip() and text.strip() != df.at[i, "text"]:
        df.at[i, "text"] = text.strip()
        df.at[i, "notes"] = (df.at[i, "notes"] + " text edited;").strip()
    df.at[i, "reviewed"] = 1
    if reviewer:
        df.at[i, "notes"] = (df.at[i, "notes"] + f" by {reviewer}").strip()
    return df


from pydantic import BaseModel  # noqa: E402  (module level so FastAPI can resolve the body model)


class Review(BaseModel):
    report_id: str
    label: int | None = None
    text: str | None = None
    reviewer: str = ""


def create_app(path: Path = CSV, language: str | None = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="Text report review")

    def queue(df):
        q = df[df.reviewed != 1]
        return q[q.language == language] if language else q

    @app.get("/next")
    def next_row(skip: str = ""):
        df = load(path)
        q = queue(df)
        skipped = [s for s in skip.split(",") if s]
        q = q[~q.report_id.isin(skipped)]
        done = int((df.reviewed == 1).sum() if not language else ((df.reviewed == 1) & (df.language == language)).sum())
        total = len(df) if not language else int((df.language == language).sum())
        if q.empty:
            return {"done": done, "total": total, "row": None}
        r = q.iloc[0]
        return {"done": done, "total": total, "row": {"report_id": r.report_id, "language": r.language,
                                                      "text": r.text, "label": int(r.label)}}

    @app.post("/review")
    def review(body: Review):
        with _lock:
            df = load(path)
            try:
                df = apply_review(df, body.report_id, body.label, body.text, body.reviewer)
            except (KeyError, ValueError) as e:
                raise HTTPException(400, str(e))
            save(df, path)
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def page():
        rubric = "".join(f"<li><b>{k}</b> {v}</li>" for k, v in RUBRIC.items())
        return PAGE.replace("{{RUBRIC}}", rubric)

    return app


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Report review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{font:18px system-ui,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;background:#f6f7f9;color:#1d2330}
.card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:20px;margin:12px 0}
#text{font-size:26px;line-height:1.5}button{font-size:18px;padding:12px 18px;margin:4px;border-radius:8px;border:1px solid #bbb;background:#fff;cursor:pointer}
.l0{background:#1a7f37;color:#fff}.l1{background:#b8860b;color:#fff}.l2{background:#d9480f;color:#fff}.l3{background:#c62828;color:#fff}
.ok{background:#1f6feb;color:#fff;font-weight:600}.muted{color:#666;font-size:14px}.cur{outline:4px solid #1f6feb}
textarea{width:100%;font-size:20px}</style></head><body>
<h2>Text severity review <span class="muted" id="prog"></span></h2>
<label class="muted">Your name <input id="who" style="font-size:16px"></label>
<div class="card"><div class="muted" id="meta"></div><div id="text">loading…</div>
<div style="margin-top:14px">Label: <b id="lab"></b></div></div>
<div><button class="ok" onclick="ok()">✔ OK (Enter)</button>
<button class="l0" onclick="setl(0)">0 stable</button><button class="l1" onclick="setl(1)">1 moderate</button>
<button class="l2" onclick="setl(2)">2 severe</button><button class="l3" onclick="setl(3)">3 critical</button>
<button onclick="edit()">✎ Edit (E)</button><button onclick="skip()">Skip (S)</button></div>
<div id="ed" style="display:none" class="card"><textarea id="ta" rows="3"></textarea>
<button class="ok" onclick="saveEdit()">Save text + OK</button></div>
<div class="card muted"><b>Rubric</b> (label by what is said, not what you guess):<ul>{{RUBRIC}}</ul>
Mark OK only if the sentence sounds like a real person in Bengaluru and the label fits the rubric.</div>
<script>
const LN={0:"0 stable",1:"1 moderate",2:"2 severe",3:"3 critical"};let row=null,skipped=[];
const who=document.getElementById('who');who.value=localStorage.getItem('who')||'';who.onchange=()=>localStorage.setItem('who',who.value);
async function load(){const r=await (await fetch('/next?skip='+skipped.join(','))).json();
document.getElementById('prog').textContent=`${r.done}/${r.total} reviewed`;row=r.row;
document.getElementById('ed').style.display='none';
if(!row){document.getElementById('text').textContent='All done 🎉';document.getElementById('lab').textContent='';return}
document.getElementById('meta').textContent=row.report_id+' · '+row.language;document.getElementById('text').textContent=row.text;
document.getElementById('lab').textContent=LN[row.label];document.getElementById('lab').className='l'+row.label}
async function send(label,text){await fetch('/review',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({report_id:row.report_id,label:label,text:text,reviewer:who.value})});load()}
function ok(){if(row)send(row.label,null)} function setl(l){if(row)send(l,null)}
function edit(){if(!row)return;document.getElementById('ed').style.display='block';const t=document.getElementById('ta');t.value=row.text;t.focus()}
function saveEdit(){send(row.label,document.getElementById('ta').value)} function skip(){if(row){skipped.push(row.report_id);load()}}
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA'||e.target.tagName==='INPUT')return;
if(e.key==='Enter')ok();else if('0123'.includes(e.key))setl(+e.key);else if(e.key==='e'||e.key==='E'){e.preventDefault();edit()}else if(e.key==='s'||e.key==='S')skip()});
load();</script></body></html>"""


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description="Review text-severity drafts in the browser")
    ap.add_argument("--language", choices=["en", "hi", "kn"])
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to let a reviewer on the same Wi-Fi open it")
    a = ap.parse_args()
    print(f"open http://{a.host if a.host != '0.0.0.0' else '<this-laptop-ip>'}:{a.port}")
    uvicorn.run(create_app(language=a.language), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
