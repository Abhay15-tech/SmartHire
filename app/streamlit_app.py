"""SmartHire web portal (Streamlit). Run: streamlit run app/streamlit_app.py"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import time
from src.parsing.resume_parser import parse_resume
from src.models.classifier import load_model as load_clf, predict_category
from src.models.recommender import JobRecommender
from src.models.skill_gap import get_skill_gap_report
from src import config

st.set_page_config(
    page_title="SmartHire | AI Career Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────── CSS ───────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Space Grotesk', sans-serif !important;
    background: #060818 !important;
    color: #e2e8f0 !important;
}

/* Animated gradient background */
.stApp {
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(109,40,217,.18) 0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 80% 90%, rgba(29,78,216,.18) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(6,8,24,1) 0%, #060818 100%) !important;
    min-height: 100vh;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container { padding: 1.5rem 3rem 4rem !important; max-width: 1400px !important; }

/* ─── Floating orbs ─── */
@keyframes floatA { 0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(40px,30px) scale(1.15);} }
@keyframes floatB { 0%,100%{transform:translate(0,0) scale(1);} 50%{transform:translate(-30px,20px) scale(1.1);} }

/* ─── HERO ─── */
.hero { text-align:center; padding:3rem 1rem 1.5rem; }
.hero-badge {
    display:inline-block;
    background:linear-gradient(135deg,rgba(139,92,246,.2),rgba(59,130,246,.15));
    border:1px solid rgba(139,92,246,.4); border-radius:50px;
    padding:5px 22px; font-size:.78rem; letter-spacing:2.5px;
    text-transform:uppercase; color:#a78bfa; margin-bottom:1.2rem;
    animation:pulseBadge 3s ease-in-out infinite;
}
@keyframes pulseBadge {
    0%,100%{box-shadow:0 0 8px rgba(139,92,246,.3);}
    50%{box-shadow:0 0 22px rgba(139,92,246,.65);}
}
.hero-title {
    font-size:clamp(2.8rem,6vw,5.5rem); font-weight:700; line-height:1.08;
    background:linear-gradient(135deg,#ffffff 0%,#a78bfa 45%,#60a5fa 100%);
    background-size:200% auto;
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation:shimmer 4s linear infinite; margin-bottom:.6rem;
}
@keyframes shimmer { to{background-position:200% center;} }
.hero-sub { font-size:1.05rem; color:#94a3b8; max-width:560px; margin:0 auto 1.4rem; line-height:1.75; }
.hero-stats { display:flex; justify-content:center; gap:3rem; flex-wrap:wrap; margin-top:.8rem; }
.hero-stat .n { font-size:1.8rem; font-weight:700; color:#a78bfa; }
.hero-stat .l { font-size:.7rem; color:#475569; text-transform:uppercase; letter-spacing:1.5px; }

/* ─── STEP BAR ─── */
.stepbar { display:flex; align-items:center; justify-content:center; max-width:600px; margin:2rem auto; }
.s-circle {
    width:44px; height:44px; border-radius:50%; display:flex; align-items:center; justify-content:center;
    font-size:.9rem; font-weight:700; border:2px solid rgba(255,255,255,.1);
    background:rgba(15,23,42,.8); color:#475569; position:relative; z-index:2;
    transition:all .4s;
}
.s-circle.active { background:linear-gradient(135deg,#7c3aed,#2563eb); border-color:transparent; color:#fff; box-shadow:0 0 22px rgba(124,58,237,.5); }
.s-circle.done   { background:linear-gradient(135deg,#059669,#10b981); border-color:transparent; color:#fff; box-shadow:0 0 14px rgba(16,185,129,.4); }
.s-lbl { font-size:.68rem; color:#475569; text-align:center; margin-top:.35rem; white-space:nowrap; }
.s-lbl.active { color:#a78bfa; } .s-lbl.done { color:#34d399; }
.s-line { flex:1; height:2px; background:rgba(255,255,255,.07); margin-bottom:1.4rem; transition:background .5s; }
.s-line.done { background:linear-gradient(90deg,#10b981,#2563eb); }
.s-wrap { display:flex; flex-direction:column; align-items:center; flex:0 0 auto; }

/* ─── UPLOAD ZONE ─── */
.upload-box {
    position:relative; overflow:hidden;
    background:linear-gradient(135deg,rgba(109,40,217,.08),rgba(29,78,216,.06));
    border:2px dashed rgba(139,92,246,.45); border-radius:24px;
    padding:3rem 2rem; text-align:center; margin-bottom:1rem;
    transition:all .35s cubic-bezier(.23,1,.32,1);
}
.upload-box::before {
    content:''; position:absolute; inset:-60%;
    background:conic-gradient(from 0deg,transparent 0%,rgba(139,92,246,.06) 30%,transparent 60%);
    animation:spinBg 10s linear infinite; pointer-events:none;
}
@keyframes spinBg { to{transform:rotate(360deg);} }
.upload-box:hover {
    border-color:rgba(139,92,246,.9);
    box-shadow:0 0 50px rgba(139,92,246,.15),inset 0 0 30px rgba(139,92,246,.06);
    transform:translateY(-4px);
}
.up-icon { font-size:3.5rem; animation:bob 2.5s ease-in-out infinite; display:block; }
@keyframes bob { 0%,100%{transform:translateY(0);} 50%{transform:translateY(-10px);} }
.up-title { font-size:1.2rem; font-weight:600; color:#e2e8f0; margin:.5rem 0 .25rem; }
.up-sub   { font-size:.85rem; color:#64748b; }
.ftype-row { display:flex; gap:.5rem; justify-content:center; margin-top:.9rem; flex-wrap:wrap; }
.ft { padding:4px 14px; border-radius:8px; font-size:.74rem; font-weight:600; letter-spacing:.5px; }
.ft-pdf  { background:rgba(239,68,68,.12); color:#f87171; border:1px solid rgba(239,68,68,.25); }
.ft-docx { background:rgba(59,130,246,.12); color:#60a5fa; border:1px solid rgba(59,130,246,.25); }
.ft-txt  { background:rgba(16,185,129,.12); color:#34d399; border:1px solid rgba(16,185,129,.25); }

/* ─── GLASSMORPHIC CARD ─── */
.gcard {
    background:linear-gradient(145deg,rgba(30,41,59,.95),rgba(15,23,42,.98));
    border:1px solid rgba(255,255,255,.07); border-radius:20px;
    padding:1.5rem 1.8rem; margin-bottom:1.1rem; position:relative; overflow:hidden;
    transition:all .4s cubic-bezier(.23,1,.32,1);
    box-shadow:0 4px 8px rgba(0,0,0,.3),0 12px 36px rgba(0,0,0,.2),inset 0 1px 0 rgba(255,255,255,.05);
}
.gcard::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(139,92,246,.45),transparent);
}
.gcard:hover {
    transform:translateY(-6px) perspective(800px) rotateX(2deg);
    border-color:rgba(139,92,246,.3);
    box-shadow:0 20px 60px rgba(0,0,0,.4),0 0 30px rgba(139,92,246,.07),inset 0 1px 0 rgba(255,255,255,.09);
}
.gcard.gold   { border-color:rgba(251,191,36,.2); }
.gcard.gold::before   { background:linear-gradient(90deg,transparent,rgba(251,191,36,.55),transparent); }
.gcard.silver { border-color:rgba(148,163,184,.2); }
.gcard.silver::before { background:linear-gradient(90deg,transparent,rgba(148,163,184,.45),transparent); }
.gcard.bronze { border-color:rgba(180,110,50,.2); }
.gcard.bronze::before { background:linear-gradient(90deg,transparent,rgba(180,110,50,.45),transparent); }

/* ─── CSS CIRCULAR SCORE ─── */
.cscore {
    position:relative; width:64px; height:64px; flex-shrink:0;
    border-radius:50%; display:flex; align-items:center; justify-content:center;
    background:conic-gradient(var(--fill-color) calc(var(--pct)*1%),rgba(255,255,255,.06) 0);
    box-shadow:0 0 16px rgba(0,0,0,.4);
}
.cscore::before {
    content:''; position:absolute;
    inset:7px; border-radius:50%;
    background:linear-gradient(145deg,rgba(30,41,59,1),rgba(15,23,42,1));
}
.cscore-num {
    position:relative; z-index:1; font-size:.72rem; font-weight:700; color:#fff; text-align:center;
}

/* ─── JOB CARD LAYOUT ─── */
.jcard-top { display:flex; align-items:flex-start; gap:1rem; }
.jcard-body { flex:1; min-width:0; }
.jcard-title { font-size:1.15rem; font-weight:600; color:#e2e8f0; }
.jcard-right { text-align:right; flex-shrink:0; }
.jcard-pct { font-size:1.4rem; font-weight:700; }
.jcard-match { font-size:.68rem; color:#64748b; }
.jmeta { display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0; }
.chip { display:inline-flex; align-items:center; gap:4px; padding:3px 10px; border-radius:8px; font-size:.76rem; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08); color:#94a3b8; }
.jdesc { font-size:.82rem; color:#64748b; line-height:1.65; margin-top:.65rem; }
.bar-bg { height:5px; border-radius:10px; background:rgba(255,255,255,.06); margin-top:.8rem; overflow:hidden; }
.bar-fill { height:100%; border-radius:10px; }
.skill-chip { display:inline-block; padding:3px 10px; background:rgba(59,130,246,.1); border:1px solid rgba(59,130,246,.22); border-radius:8px; font-size:.73rem; color:#93c5fd; margin:2px; }

/* ─── CATEGORY CARD ─── */
.cat-card {
    display:flex; align-items:center; gap:1.5rem;
    padding:1.5rem 2rem;
    background:linear-gradient(135deg,rgba(139,92,246,.14),rgba(59,130,246,.1));
    border:1px solid rgba(139,92,246,.28); border-radius:18px;
    margin-bottom:1.2rem; position:relative; overflow:hidden;
}
.cat-card::after {
    content:''; position:absolute; top:-40%; right:-5%;
    width:200px; height:200px;
    background:radial-gradient(circle,rgba(139,92,246,.14) 0%,transparent 70%);
    pointer-events:none;
}
.cat-ico { font-size:3.2rem; flex-shrink:0; }
.cat-lbl { font-size:.7rem; color:#94a3b8; letter-spacing:2px; text-transform:uppercase; margin-bottom:.25rem; }
.cat-name { font-size:1.55rem; font-weight:700; color:#e2e8f0; }
.cat-pill {
    display:inline-block; margin-top:.45rem;
    padding:3px 14px; background:linear-gradient(90deg,#7c3aed,#2563eb);
    border-radius:30px; font-size:.73rem; font-weight:600; color:#fff;
}

/* ─── STATS MINI ─── */
.stats2 { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin-top:.8rem; }
.stat2 { text-align:center; padding:.9rem; background:rgba(255,255,255,.03); border-radius:12px; border:1px solid rgba(255,255,255,.05); }
.stat2 .n { font-size:1.6rem; font-weight:700; }
.stat2 .l { font-size:.68rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-top:.1rem; }

/* ─── INFO TIP ─── */
.itip { background:rgba(59,130,246,.07); border-left:3px solid #2563eb; border-radius:0 12px 12px 0; padding:10px 14px; font-size:.8rem; color:#94a3b8; margin-top:.9rem; }
.itip.purple { background:rgba(139,92,246,.07); border-left-color:#7c3aed; }

/* ─── SECTION HEADER ─── */
.sh { display:flex; align-items:center; gap:.8rem; margin-bottom:1.1rem; }
.sh .ico { font-size:1.4rem; }
.sh h3 { font-size:1.2rem; font-weight:600; color:#e2e8f0; margin:0; }
.sh .cnt { margin-left:auto; background:rgba(139,92,246,.14); border:1px solid rgba(139,92,246,.28); border-radius:20px; padding:2px 13px; font-size:.75rem; color:#a78bfa; }

/* ─── MEDAL ─── */
.medal { font-size:1.3rem; margin-right:.2rem; }

/* ─── LOADING ─── */
.wave { display:flex; gap:7px; align-items:center; justify-content:center; padding:.8rem; }
.wd { width:11px; height:11px; border-radius:50%; background:#7c3aed; animation:waveDot 1.3s ease-in-out infinite; }
.wd:nth-child(2){animation-delay:.18s;background:#4f46e5;}
.wd:nth-child(3){animation-delay:.36s;background:#2563eb;}
@keyframes waveDot { 0%,60%,100%{transform:translateY(0);opacity:.4;} 30%{transform:translateY(-13px);opacity:1;} }

/* ─── FADE ANIMATIONS ─── */
@keyframes fadeUp { from{opacity:0;transform:translateY(22px);} to{opacity:1;transform:translateY(0);} }
.fu  { animation:fadeUp .55s ease forwards; }
.fu2 { animation:fadeUp .55s ease .12s both; }
.fu3 { animation:fadeUp .55s ease .25s both; }

/* ─── STREAMLIT FILE UPLOADER TWEAKS ─── */
[data-testid="stFileUploaderDropzone"] {
    background:rgba(15,23,42,.4) !important;
    border:1.5px dashed rgba(139,92,246,.35) !important;
    border-radius:14px !important;
}
</style>
"""

MEDALS = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
RANK_CLS = ["gold", "silver", "bronze", "", ""]

SCORE_GRADIENTS = {
    "high": ("linear-gradient(135deg,#f59e0b,#ef4444)", "#f59e0b"),
    "mid":  ("linear-gradient(135deg,#2563eb,#7c3aed)", "#6366f1"),
    "low":  ("linear-gradient(135deg,#059669,#10b981)", "#10b981"),
}
def sgrd(pct):
    if pct >= 60: return SCORE_GRADIENTS["high"]
    if pct >= 35: return SCORE_GRADIENTS["mid"]
    return SCORE_GRADIENTS["low"]

CAT_ICONS = {
    "Data Science":"📊","Python Developer":"🐍","Web Designing":"🎨","HR":"👥",
    "Advocate":"⚖️","Arts":"🎭","Mechanical Engineer":"⚙️","Sales":"💼",
    "Health and fitness":"🏋️","Civil Engineer":"🏗️","Java Developer":"☕",
    "Business Analyst":"📈","SAP Developer":"💻","Automation Testing":"🤖",
    "Electrical Engineering":"⚡","Operations Manager":"🏭","Chef":"👨‍🍳",
    "Blockchain":"🔗","Teacher":"📚","Network Security Engineer":"🔒",
    "DotNet Developer":"🖥️","Database":"🗄️","Hadoop":"🐘",
    "ETL Developer":"🔄","DevOps Engineer":"🛠️","PMO":"📋",
}

@st.cache_resource
def load_models():
    clf_model, clf_vec = load_clf(config.MODELS_DIR / "resume_classifier.pkl")
    recommender = JobRecommender.load(config.MODELS_DIR / "job_recommender.pkl")
    return clf_model, clf_vec, recommender

def stepbar(step):
    labels = ["Upload Resume", "AI Analysis", "Results"]
    parts = []
    for i, lbl in enumerate(labels):
        n = i + 1
        if step > n:   cls, txt = "done",   "✓"
        elif step == n: cls, txt = "active", str(n)
        else:           cls, txt = "",       str(n)
        lbl_cls = "done" if step > n else ("active" if step == n else "")
        if i > 0:
            line_cls = "done" if step > i else ""
            parts.append(f'<div class="s-line {line_cls}"></div>')
        parts.append(f'''
        <div class="s-wrap">
          <div class="s-circle {cls}">{txt}</div>
          <div class="s-lbl {lbl_cls}">{lbl}</div>
        </div>''')
    st.markdown(f'<div class="stepbar">{"".join(parts)}</div>', unsafe_allow_html=True)

def job_card(job, rank, pct):
    grad, hex_c = sgrd(pct)
    medal  = MEDALS[rank]
    rcls   = RANK_CLS[rank]

    title   = str(job.get("title", "N/A"))
    company = str(job.get("company", "N/A"))
    loc     = str(job.get("location", "N/A"))
    exp     = str(job.get("experience", "N/A"))
    desc    = str(job.get("description", ""))[:230]

    raw_sk = str(job.get("skills", ""))
    skills_html = ""
    if raw_sk and raw_sk.lower() != "nan":
        for sk in raw_sk.split(",")[:6]:
            sk = sk.strip()
            if sk:
                skills_html += f'<span class="skill-chip">{sk}</span>'

    st.markdown(f"""
<div class="gcard {rcls} fu" style="animation-delay:{rank*.1}s">
  <div class="jcard-top">
    <div class="cscore" style="--pct:{pct};--fill-color:{hex_c}">
      <div class="cscore-num">{pct}%</div>
    </div>
    <div class="jcard-body">
      <div class="jcard-title"><span class="medal">{medal}</span>{title}</div>
      <div class="jmeta">
        <span class="chip">🏢 {company}</span>
        <span class="chip">📍 {loc}</span>
        <span class="chip">⏳ {exp}</span>
      </div>
    </div>
    <div class="jcard-right">
      <div class="jcard-pct" style="color:{hex_c}">{pct}%</div>
      <div class="jcard-match">match</div>
    </div>
  </div>
  <div class="bar-bg"><div class="bar-fill" style="width:{pct}%;background:{grad}"></div></div>
  {f'<div style="margin-top:.6rem">{skills_html}</div>' if skills_html else ""}
  <div class="jdesc">{desc}{"…" if len(str(job.get("description",""))) > 230 else ""}</div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────── MAIN ───────────────────────────
def main():
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    # ── Hero ──
    st.markdown("""
<div class="hero fu">
  <div class="hero-badge">✦ &nbsp; AI-Powered Career Engine &nbsp; ✦</div>
  <div class="hero-title">SmartHire</div>
  <p class="hero-sub">Upload your resume — our AI predicts your ideal role,
  finds your top job matches, and surfaces the skills that get you hired.</p>
  <div class="hero-stats">
    <div class="hero-stat"><div class="n">5+</div><div class="l">Job Matches</div></div>
    <div class="hero-stat"><div class="n">ML</div><div class="l">Powered</div></div>
    <div class="hero-stat"><div class="n">3s</div><div class="l">Analysis</div></div>
    <div class="hero-stat"><div class="n">TF-IDF</div><div class="l">Matching</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Load models ──
    try:
        clf_model, clf_vec, recommender = load_models()
        models_ok = True
    except Exception:
        models_ok = False

    if not models_ok:
        st.markdown("""
<div class="gcard" style="border-color:rgba(239,68,68,.3);text-align:center;padding:2.5rem">
  <div style="font-size:3rem;margin-bottom:1rem">⚠️</div>
  <div style="font-size:1.2rem;font-weight:600;color:#f87171;margin-bottom:.5rem">Models Not Found</div>
  <div style="color:#64748b">Run <code style="background:rgba(239,68,68,.1);padding:2px 8px;border-radius:6px;color:#f87171">python -m src.train</code> from the project root first.</div>
</div>""", unsafe_allow_html=True)
        return

    # ── Step bar placeholder ──
    step_ph = st.empty()
    step_ph.markdown('<div class="stepbar">' + "" + "</div>", unsafe_allow_html=True)
    with step_ph.container():
        stepbar(1)

    # ── Upload UI ──
    st.markdown("""
<div class="upload-box fu2">
  <span class="up-icon">📄</span>
  <div class="up-title">Drop Your Resume Here</div>
  <div class="up-sub">or click the button below to browse files</div>
  <div class="ftype-row">
    <span class="ft ft-pdf">PDF</span>
    <span class="ft ft-docx">DOCX</span>
    <span class="ft ft-txt">TXT</span>
  </div>
</div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    if not uploaded:
        st.markdown('<div class="itip" style="max-width:600px;margin:0 auto">💡 Your resume is processed entirely on your machine — nothing is sent to external servers.</div>', unsafe_allow_html=True)
        return

    # ── Step 2 ──
    with step_ph.container():
        stepbar(2)

    st.markdown("""
<div class="gcard fu" style="text-align:center;padding:1.8rem">
  <div class="wave"><div class="wd"></div><div class="wd"></div><div class="wd"></div></div>
  <div style="color:#94a3b8;font-size:.9rem;margin-top:.3rem">🧠 Parsing resume &amp; running AI analysis…</div>
</div>""", unsafe_allow_html=True)

    with st.spinner(""):
        time.sleep(0.6)
        try:
            resume_text = parse_resume(uploaded, uploaded.name)
            category    = predict_category(clf_model, clf_vec, resume_text)
            top_jobs    = recommender.recommend(resume_text, top_n=5)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return

    # ── Step 3 ──
    with step_ph.container():
        stepbar(3)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([4, 6], gap="large")

    with left:
        # Match cluster/role family
        matched_cluster = top_jobs[0].get("cluster_label", "Not specified") if top_jobs else "Not specified"

        # Category
        icon = CAT_ICONS.get(category, "💡")
        st.markdown(f"""
<div class="cat-card fu">
  <div class="cat-ico">{icon}</div>
  <div>
    <div class="cat-lbl">AI Predicted Role</div>
    <div class="cat-name">{category}</div>
    <div style="font-size:0.8rem; color:#60a5fa; margin-top:0.25rem;">Role Family: {matched_cluster}</div>
    <span class="cat-pill">✨ Best Match</span>
  </div>
</div>""", unsafe_allow_html=True)

        # Resume stats
        words = len(resume_text.split())
        chars = len(resume_text)
        st.markdown(f"""
<div class="gcard fu2">
  <div class="sh"><span class="ico">📋</span><h3>Resume Stats</h3></div>
  <div class="stats2">
    <div class="stat2"><div class="n" style="color:#a78bfa">{words:,}</div><div class="l">Words</div></div>
    <div class="stat2"><div class="n" style="color:#60a5fa">{chars:,}</div><div class="l">Characters</div></div>
  </div>
  <div class="itip">💡 Best ATS results: 400–800 words with keyword-rich content.</div>
</div>""", unsafe_allow_html=True)

        # Skill Gap Report
        try:
            skills_report = get_skill_gap_report(
                resume_text, category, recommender.job_corpus, role_col="category"
            )
            
            gap_badges = ""
            possess_badges = ""
            for item in skills_report:
                skill = item["skill"]
                pct = item["importance"]
                if item["present"]:
                    possess_badges += f'<span class="skill-chip" style="background:rgba(16,185,129,.07); border-color:rgba(16,185,129,.25); color:#34d399">✓ {skill}</span>'
                else:
                    gap_badges += f'<span class="skill-chip" style="background:rgba(139,92,246,.07); border-color:rgba(139,92,246,.25); color:#a78bfa">⚡ {skill} ({pct}% demand)</span>'
            
            st.markdown(f"""
<div class="gcard fu3">
  <div class="sh"><span class="ico">🎯</span><h3>Skills to Improve</h3></div>
  <div style="margin-bottom:0.7rem">
    <div style="font-size:0.75rem; color:#a78bfa; margin-bottom:0.35rem; text-transform:uppercase; letter-spacing:1px">Missing Skills (Target Gaps)</div>
    {gap_badges if gap_badges else '<div style="font-size:0.8rem; color:#64748b">No gaps identified!</div>'}
  </div>
  <div>
    <div style="font-size:0.75rem; color:#34d399; margin-bottom:0.35rem; text-transform:uppercase; letter-spacing:1px">Strengths (Already Present)</div>
    {possess_badges if possess_badges else '<div style="font-size:0.8rem; color:#64748b">None identified.</div>'}
  </div>
  <div class="itip purple">💡 Boost your match score by adding missing skills back to your profile or projects list!</div>
</div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error computing skills gap: {e}")

    with right:
        st.markdown(f"""
<div class="sh fu">
  <span class="ico">🏆</span>
  <h3>Top Job Matches</h3>
  <span class="cnt">{len(top_jobs)} results</span>
</div>""", unsafe_allow_html=True)

        for i, job in enumerate(top_jobs):
            pct = max(1, int(job["similarity"] * 100))
            job_card(job, i, pct)

        st.markdown("""
<div class="itip purple fu">
  🔍 Scores are computed via TF-IDF cosine similarity between your resume text and job descriptions.
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
