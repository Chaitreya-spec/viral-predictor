"""
Streamlit upload UI: drag in a video → get a viral-potential verdict + reasons.

Launch:  streamlit run app.py
(opens in your browser at http://localhost:8501)
"""
import tempfile, os
import streamlit as st
from predict import score_video

st.set_page_config(page_title="Will It Go Viral?", page_icon="🎬")

st.title("🎬 Will It Go Viral?")
st.caption("Upload a video and see whether the model thinks it will over- or "
           "under-perform its channel's typical views.")

uploaded = st.file_uploader(
    "Drop a video here", type=["mp4", "mov", "mkv", "webm", "m4v"])
title = st.text_input("Video title", placeholder="e.g. I climbed the Alps solo")
niche = st.selectbox("Niche", ["mountain", "racing", "travel"])
channel_median = st.number_input(
    "Your channel's typical view count", min_value=1, value=10000, step=500,
    help="The model predicts performance relative to this baseline.")

if st.button("Predict", type="primary", disabled=uploaded is None):
    # save the upload to a temp file so the extractor can read it
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

    try:
        with st.spinner("Analysing video (extracting features + scoring)…"):
            r = score_video(path, title, channel_median, niche)
    finally:
        os.unlink(path)

    lift, mult, verdict = r["lift"], r["multiplier"], r["verdict"]

    # headline verdict, colour-coded
    if lift > 0.7:
        st.success(f"### {verdict}")
    elif lift > -0.2:
        st.info(f"### {verdict}")
    else:
        st.warning(f"### {verdict}")

    c1, c2 = st.columns(2)
    c1.metric("Predicted lift", f"{lift:+.2f}")
    c2.metric("vs channel's normal", f"~{mult:.1f}×")

    st.subheader("What drove this")
    for label, c in r["factors"][:6]:
        arrow = "🟢 helps" if c > 0 else "🔴 hurts"
        st.write(f"{arrow} — **{label}**  ({c:+.2f})")

    st.caption("Model: LightGBM on ~500 videos. Trained on relative performance "
               "(lift), so channel size is controlled for, not rewarded.")
