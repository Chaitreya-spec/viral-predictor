"""
Web app backend: serves the UI and exposes POST /predict.

Run from the project root:
    uvicorn src.api:app --reload
Then open http://localhost:8000

Scoring reuses predict.score_video (which extracts features in a subprocess,
so PyTorch and LightGBM never share a process — no segfault).
"""
import os, sys, tempfile, pathlib
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, os.path.dirname(__file__))   # make sibling modules importable
from predict import score_video

app = FastAPI(title="Will It Go Viral?")
WEB = pathlib.Path(__file__).parent / "web"


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text()


@app.post("/predict")
async def predict(video: UploadFile = File(...),
                  title: str = Form(""),
                  niche: str = Form(""),
                  channel_median: float = Form(10000)):
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await video.read())
        path = tmp.name
    try:
        r = score_video(path, title, channel_median, niche)
    finally:
        os.unlink(path)

    return JSONResponse({
        "verdict":    r["verdict"],
        "lift":       round(r["lift"], 2),
        "multiplier": round(r["multiplier"], 1),
        "factors":    [{"label": lbl, "value": round(c, 2)}
                       for lbl, c in r["factors"][:6]],
    })
