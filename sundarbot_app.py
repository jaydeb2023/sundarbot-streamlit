"""
SundarBot — Streamlit MVP (single file, extended)
-------------------------------------------------
Adds:
- Weather (rain + 5-day forecast via OpenWeatherMap; set OWM_API_KEY env var)
- Best Places recommender (season/time/interest-aware)
- Add Places (Hotels + Shops/Vendors) with map picker; persists to places.csv
- Nearby finder (radius/category) + map markers
- Events / important dates
- Image gallery (remote URLs or local assets)
- NEW: Tour Companies (list-only): public add form + searchable table (tour_companies.csv)
- Original chat, KB, export, and portfolio preserved

Run:
1) pip install streamlit scikit-learn pandas numpy folium streamlit-folium reportlab requests python-dateutil
2) streamlit run sundarbot_app.py
"""

import os
import io
import json
import time
from typing import List, Dict, Any
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from folium import Map, Marker, Popup, Icon
from streamlit_folium import st_folium

# ---------- Optional weather deps ----------
import requests
from dateutil import tz

# =========================================================
# Config / constants
# =========================================================
PAGE_ICON = "🌿"            # app icon
PRIMARY_HEX = "#0f766e"     # teal-green
ACCENT_HEX = "#eab308"      # sandy/gold
DEFAULT_CENTER = (22.000, 88.700)  # approx Sundarban center (lat, lon)
OWM_KEY = os.getenv("dd3675418a757617fb78d0285f3fd1af")  # OpenWeatherMap key (optional)

# ---------- Data files ----------
SUBMISSIONS_CSV = "submissions.csv"
PLACES_CSV = "places.csv"  # user-added hotels/shops/vendors
PLACE_FIELDS = [
    "name","cat","lat","lon","address","phone","hours","website","notes","image_url","timestamp"
]

# NEW: Tour company CSV
TOUR_CSV = "tour_companies.csv"
TOUR_FIELDS = [
    "company_name","contact_person","phone","email","website","address","license_no",
    "services","notes","timestamp"
]

# =========================================================
# Knowledge Base (seed; extend freely)
# =========================================================
KB_DOCS = [
    {
        "id": "wildlife_tiger",
        "title": "Royal Bengal Tiger",
        "section": "Wildlife",
        "text": (
            "The Sundarban is home to the Royal Bengal Tiger, adapted to a mangrove habitat. "
            "Tigers here often swim between islands, and their prey includes chital deer, wild boar, and fish. "
            "Conservation focuses on minimizing human-wildlife conflict and protecting core tiger habitats."
        ),
        "refs": [{"label": "IUCN Tiger", "url": "https://www.iucnredlist.org/species/15955/50659951"}],
    },
    {
        "id": "mangroves_ecology",
        "title": "Mangrove Ecology",
        "section": "Mangroves",
        "text": (
            "Sundarban mangroves, dominated by Heritiera fomes and Avicennia spp., buffer storms, store carbon, "
            "and provide nursery habitat for fish and crustaceans. Salinity gradients shape species distributions."
        ),
        "refs": [{"label": "UNESCO Sundarbans", "url": "https://whc.unesco.org/en/list/452/"}],
    },
    {
        "id": "birds",
        "title": "Birdlife",
        "section": "Wildlife",
        "text": (
            "Over 300 bird species are recorded, including kingfishers, egrets, herons, raptors, and migratory shorebirds. "
            "Birdwatching is best during winter migration, at river edges and mudflats during low tide."
        ),
        "refs": [],
    },
    {
        "id": "communities",
        "title": "Local Communities",
        "section": "Communities",
        "text": (
            "Local livelihoods include fishing, honey collection, and small-scale farming. Eco-tourism initiatives create alternative income and raise conservation awareness. "
            "Respect local customs, hire licensed guides, and support community-run homestays where possible."
        ),
        "refs": [],
    },
    {
        "id": "eco_tips",
        "title": "Eco-friendly Tips",
        "section": "Tips",
        "text": (
                "Carry back all waste, use reef-safe sunscreen, keep noise low on boats, and maintain a safe distance from wildlife. "
                "Choose certified operators and respect protected area rules and seasonal closures."
        ),
        "refs": [],
    },
]

# =========================================================
# Portfolio (replace URLs with real links)
# =========================================================
PORTFOLIO = [
    {
        "name": "Camera Trap Wildlife Detection (YOLOv8)",
        "url": "https://github.com/<your-user>/yolov8-camera-trap",
        "desc": "End-to-end pipeline: dataset curation, training, eval, and on-device inference.",
        "tags": ["Computer Vision","YOLOv8","MLOps"],
    },
    {
        "name": "Drone-based Mangrove Canopy Mapping",
        "url": "https://github.com/<your-user>/mangrove-canopy-mapping",
        "desc": "Orthomosaic + canopy indices (NDVI/NDRE).",
        "tags": ["Remote Sensing","NDVI","GIS"],
    },
    {
        "name": "Poaching Risk Forecast (Time Series)",
        "url": "https://github.com/<your-user>/poaching-risk-timeseries",
        "desc": "Spatio-temporal risk maps & ranked hotspots for patrol planning.",
        "tags": ["Time Series","Forecasting","Geo"],
    },
]

# =========================================================
# Retrieval over KB (TF-IDF)
# =========================================================
class Retriever:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.corpus = [d["title"] + "\n" + d["text"] for d in docs]
        self.doc_vectors = self.vectorizer.fit_transform(self.corpus)

    def search(self, query: str, top_k: int = 3):
        if not query:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_vectors).ravel()
        idxs = sims.argsort()[::-1][:top_k]
        results = []
        for i in idxs:
            d = self.docs[i]
            results.append({
                "id": d["id"],
                "title": d["title"],
                "section": d["section"],
                "snippet": d["text"][:300] + ("..." if len(d["text"])>300 else ""),
                "score": float(sims[i]),
                "refs": d.get("refs", []),
            })
        return results

@st.cache_resource
def get_retriever():
    return Retriever(KB_DOCS)

retriever = get_retriever()

# =========================================================
# Simple LLM gateway (stub; replace with your model)
# =========================================================
SYSTEM_PROMPT = (
    "You are 'SundarBot', an AI guide for the Sundarban ecosystem. Be accurate, friendly, and conservation-first. "
    "Prioritize safety, local regulations, and eco-friendly practices. If unsure, say so and suggest trusted sources."
)

def llm_generate(system_prompt: str, history: List[Dict[str, str]], user_query: str, kb_context: str) -> str:
    # Replace with a real model call if you like.
    answer = (
        "Thanks for asking! Here's what I found related to your question.\n\n" +
        kb_context +
        "\n\nIf you need routes, birding spots, or eco-friendly tips for your itinerary, tell me your trip dates and interests."
    )
    return answer

# =========================================================
# Weather (OpenWeatherMap One Call)
# =========================================================
@st.cache_data(ttl=60*30)  # cache 30 min
def fetch_weather(lat: float, lon: float, units: str = "metric"):
    if not OWM_KEY:
        return {"error": "Missing OWM_API_KEY"}
    params = {
        "lat": lat, "lon": lon,
        "appid": OWM_KEY,
        "units": units,
        "exclude": "minutely,alerts",
    }
    url = "https://api.openweathermap.org/data/2.5/onecall"
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def render_weather_panel(lat: float, lon: float, tz_name="Asia/Kolkata"):
    data = fetch_weather(lat, lon)
    if "error" in data:
        st.warning("Set OWM_API_KEY to enable live weather.")
        return
    local_tz = tz.gettz(tz_name)
    current = data.get("current", {})
    daily = data.get("daily", [])[:5]

    st.subheader("🌦 Weather & Rain Forecast")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        st.metric("Now", f"{round(current.get('temp', 0))}°C", help=current.get("weather", [{}])[0].get("description",""))
    with colB:
        pop = current.get("pop")
        if pop is None:
            pop = 0
        else:
            pop = int(round(pop*100))
        st.metric("Rain chance (now)", f"{pop}%")
    with colC:
        wind = current.get("wind_speed", 0)
        st.metric("Wind", f"{round(wind)} m/s")
    with colD:
        hum = current.get("humidity", 0)
        st.metric("Humidity", f"{hum}%")

    st.write("**Next 5 days**")
    for d in daily:
        dt_local = datetime.fromtimestamp(d["dt"], tz=local_tz).strftime("%a %d %b")
        tmin = round(d["temp"]["min"]); tmax = round(d["temp"]["max"])
        pop  = int(round(d.get("pop", 0)*100))
        rain = d.get("rain", 0)
        desc = d.get("weather",[{}])[0].get("description","")
        st.markdown(f"- **{dt_local}** — {tmin}–{tmax}°C • rain chance **{pop}%** • rain {rain} mm • {desc}")

# =========================================================
# Places: built-in + user-added (CSV)
# =========================================================
def load_user_places() -> pd.DataFrame:
    try:
        df = pd.read_csv(PLACES_CSV)
        missing = [c for c in PLACE_FIELDS if c not in df.columns]
        for m in missing: df[m] = ""
        return df[PLACE_FIELDS]
    except Exception:
        return pd.DataFrame(columns=PLACE_FIELDS)

def save_user_place(row: dict):
    df = load_user_places()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(PLACES_CSV, index=False)

def combined_places(default_list: list[dict]) -> pd.DataFrame:
    user = load_user_places()
    base = pd.DataFrame(default_list)
    for c in PLACE_FIELDS:
        if c not in base.columns:
            base[c] = ""
    return pd.concat([base[PLACE_FIELDS], user[PLACE_FIELDS]], ignore_index=True)

# default local places (edit/extend)
LOCAL_PLACES = [
    {"name":"Sajnekhali Permit Counter","cat":"permit","lat":22.109,"lon":88.797,
     "address":"Sajnekhali","phone":"","hours":"8:00–16:00","website":"","notes":"Official permits & basic info","image_url":"","timestamp":""},
    {"name":"Eco Homestay A","cat":"hotel","lat":22.070,"lon":88.750,
     "address":"Near Sajnekhali","phone":"+91-0000-000000","hours":"24h","website":"","notes":"Family-friendly; basic amenities","image_url":"","timestamp":""},
    {"name":"Boat Supplies Shop","cat":"shop","lat":22.085,"lon":88.720,
     "address":"Main Jetty lane","phone":"+91-0000-000111","hours":"7:00–18:00","website":"","notes":"Rain gear, life jackets, snacks","image_url":"","timestamp":""},
    {"name":"Local Guide — Mr. Roy","cat":"vendor","lat":22.082,"lon":88.740,
     "address":"Marketplace","phone":"+91-0000-000222","hours":"By appt","website":"","notes":"Licensed guide; birding & creeks","image_url":"","timestamp":""},
]

# =========================================================
# POIs + Recommender
# =========================================================
POIS = [
    {
        "name": "Sajnekhali Interpretation Centre",
        "lat": 22.109, "lon": 88.797, "type": "Visitor",
        "tags": ["beginner","permit","education","family"],
        "best_seasons": ["winter","post-monsoon"],
        "best_times": ["morning","afternoon"],
        "tips": "Permits/info; easy start; nearby watchtowers & creeks.",
    },
    {
        "name": "Dobanki Watch Tower",
        "lat": 22.090, "lon": 88.666, "type": "Wildlife",
        "tags": ["birding","canopy","scenic"],
        "best_seasons": ["winter","pre-monsoon"],
        "best_times": ["morning","late-afternoon"],
        "tips": "Canopy walk; kingfishers, herons; quiet creeks.",
    },
    {
        "name": "Netidhopani Watch Tower",
        "lat": 21.832, "lon": 88.530, "type": "Wildlife",
        "tags": ["advanced","restricted","tiger-habitat"],
        "best_seasons": ["winter"],
        "best_times": ["morning"],
        "tips": "Restricted access; confirm permits and tides.",
    },
    {
        "name": "Bhagabatpur Crocodile Project",
        "lat": 21.712, "lon": 88.550, "type": "Conservation",
        "tags": ["family","education","crocodile"],
        "best_seasons": ["winter","post-monsoon"],
        "best_times": ["morning","midday"],
        "tips": "Estuarine crocodile rearing; good for families.",
    },
]

def recommend_pois(interests: list[str], season: str | None = None, time_of_day: str | None = None, k: int = 5):
    """Tiny rules-based recommender using tag matches + seasonal/time boosts."""
    season = (season or "").lower()
    time_of_day = (time_of_day or "").lower()
    wants = {w.strip().lower() for w in interests}

    scored = []
    for p in POIS:
        score = 0
        tags = {t.lower() for t in p.get("tags", [])} | {p.get("type","").lower()}
        score += 2 * len(wants & tags)
        if season and season in [s.lower() for s in p.get("best_seasons", [])]:
            score += 2
        if time_of_day and time_of_day in [t.lower() for t in p.get("best_times", [])]:
            score += 1
        if "restricted" in tags and "advanced" not in wants:
            score -= 1
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [dict(p, score=s) for s, p in scored[:k] if s > 0] or [dict(scored[0][1], score=scored[0][0])]
    return top

# =========================================================
# Utilities (distance)
# =========================================================
from math import radians, cos, sin, asin, sqrt

def haversine_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0
    dlat, dlon = radians(b_lat-a_lat), radians(b_lon-a_lon)
    lat1, lat2 = radians(a_lat), radians(b_lat)
    h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2*R*asin(sqrt(h))

# =========================================================
# Tour Companies: load/save
# =========================================================
def load_tours() -> pd.DataFrame:
    try:
        df = pd.read_csv(TOUR_CSV)
        # ensure all fields exist
        missing = [c for c in TOUR_FIELDS if c not in df.columns]
        for m in missing: df[m] = ""
        return df[TOUR_FIELDS]
    except Exception:
        return pd.DataFrame(columns=TOUR_FIELDS)

def save_tour(row: dict):
    df = load_tours()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(TOUR_CSV, index=False)

# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="SundarBot — Sundarban Eco-Guide", page_icon=PAGE_ICON, layout="wide")

# ----- CSS -----
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #f0fdf4 0%, #ecfeff 100%);
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial;
    }}
    .sundar-hero {{
        background: url('https://upload.wikimedia.org/wikipedia/commons/6/6b/Royal_Bengal_Tiger_in_the_mangrove_habitat_of_Sundarban_National_Park%2C_West_Bengal%2C_India.jpg') no-repeat;
        background-size: cover;
        background-position: 50% 85%;
        min-height: 430px;
        border-radius: 18px; padding: 24px; color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        position: relative; overflow: hidden;
    }}
    .sundar-hero::after {{
        content: ""; position: absolute; inset: 0;
        background: rgba(0,0,0,0.24);
    }}
    .sundar-hero > * {{ position: relative; z-index: 1; }}
    .pill {{ display: inline-block; padding: 6px 12px; border-radius: 999px; background: {PRIMARY_HEX}; color: white; font-weight: 600; }}
    .card {{ background: white; border-radius: 16px; padding: 16px; box-shadow: 0 6px 20px rgba(0,0,0,0.08); }}
    .muted {{ color: #475569; }}
    .btn-primary button {{ background: {PRIMARY_HEX} !important; border-color: {PRIMARY_HEX} !important; }}
    a, .stMarkdown a {{ color: {PRIMARY_HEX}; }}
    @media (max-width: 768px) {{
        .sundar-hero {{ min-height: 320px; background-position: 50% 80%; }}
    }}
    /* tighten page gutters so map can stretch */
    .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----- Hero -----
st.markdown(
    """
    <div class="sundar-hero">
      <div class="pill">SundarBot</div>
      <h1 style="margin: 8px 0 4px">Your Sundarban Eco-Guide</h1>
      <p style="max-width: 900px">Learn about mangroves, wildlife, conservation and eco-tourism. Ask about routes, permits, or birding hotspots. Submit sightings to help science, and browse my AI portfolio.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----- Sidebar (Portfolio + Nearby) -----
with st.sidebar:
    st.header("👨‍💻 Portfolio")
    for p in PORTFOLIO:
        st.markdown(f"- [{p['name']}]({p['url']})")
        if p.get("desc"): st.caption(p["desc"])
    st.divider()

    st.markdown("### 📍 Nearby (beta)")
    base_lat = st.number_input("Base lat", value=float(DEFAULT_CENTER[0]), format="%.6f")
    base_lon = st.number_input("Base lon", value=float(DEFAULT_CENTER[1]), format="%.6f")
    cat = st.multiselect("Category", ["permit","hotel","shop","vendor","restaurant","medical"], default=["hotel","shop","vendor"])
    radius = st.slider("Radius (km)", 1, 50, 15)

    ALL_PLACES_DF = combined_places(LOCAL_PLACES)
    nearby = []
    for _, p in ALL_PLACES_DF.iterrows():
        if p["cat"] in cat:
            d = haversine_km(base_lat, base_lon, float(p["lat"]), float(p["lon"]))
            if d <= radius:
                row = p.to_dict()
                row["distance_km"] = round(d,1)
                nearby.append(row)
    if nearby:
        st.caption("Results (nearest first):")
        for p in sorted(nearby, key=lambda x: x["distance_km"]):
            st.markdown(
                f"**{p['name']}** — {p['cat']} • {p['distance_km']} km  \n"
                f"{p.get('address','')}  \n"
                f"{'📞 '+p['phone'] if p.get('phone') else ''} "
                f"{' • ⏰ '+p['hours'] if p.get('hours') else ''}  "
                f"{' • 🔗 ['+p['website']+']('+p['website']+')' if p.get('website') else ''}"
            )
    else:
        st.caption("No matches in selected radius.")

# ----- Tabs -----
tabs = st.tabs([
    "💬 Chat", "🗺️ Map", "📝 Submit",
    "⭐ Best Places", "🌧 Weather", "🖼 Gallery", "➕ Add Places",
    "🏝️ Tour Companies", "📚 Knowledge Base"
])

# =========================================================
# Tab 1: Chat
# =========================================================
with tabs[0]:
    if "chat" not in st.session_state: st.session_state.chat = []
    def add_chat(role: str, content: str): st.session_state.chat.append({"role": role, "content": content})

    # show history
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_query = st.chat_input("Ask about wildlife, routes, safety, or eco-tourism…")
    if user_query:
        add_chat("user", user_query)

        # retrieve
        hits = retriever.search(user_query, top_k=3)
        kb_context = "\n\n".join([f"- {h['title']}: {h['snippet']}" for h in hits]) or "(No direct KB match; answering generally.)"

        # base answer
        answer = llm_generate(SYSTEM_PROMPT, st.session_state.chat, user_query, kb_context)

        # naive recommend intent
        uq = user_query.lower()
        if any(w in uq for w in ["recommend","suggest","best place","where should i go","itinerary"]):
            season = None
            for s in ["winter","summer","monsoon","pre-monsoon","post-monsoon"]:
                if s in uq: season = s
            time_hint = None
            for t in ["morning","afternoon","late-afternoon","evening"]:
                if t in uq: time_hint = t

            # weather-aware tweak (optional)
            rain_prob_now = None
            if OWM_KEY:
                try:
                    wx = fetch_weather(*DEFAULT_CENTER)
                    hourly = (wx.get("hourly") or [{}])[0]
                    if "pop" in hourly:
                        rain_prob_now = int(round(hourly["pop"]*100))
                    if (rain_prob_now is not None) and rain_prob_now >= 60 and (not season):
                        season = "monsoon"
                except Exception:
                    pass

            interests = [kw for kw in ["birding","family","beginner","education","canopy","crocodile","tiger","advanced"] if kw in uq] or ["birding"]
            recs = recommend_pois(interests, season=season, time_of_day=time_hint, k=3)
            st.session_state.last_recs = recs  # store for map highlight

            tip = f" (rain chance {rain_prob_now}% — choose sheltered spots)" if rain_prob_now is not None else ""
            lines = [f"**{r['name']}** — {r['type']}  \n*Why:* {r.get('tips','')}" for r in recs]
            answer += f"\n\n**Recommended places{tip}:**\n" + "\n".join(f"- {x}" for x in lines)

        add_chat("assistant", answer)
        with st.chat_message("assistant"):
            st.markdown(answer)
            if hits:
                with st.expander("References & related"):
                    for h in hits:
                        st.write(f"**{h['title']}** — {h['section']} (score: {h['score']:.2f})")
                        for r in h.get("refs", []):
                            st.markdown(f"- [{r['label']}]({r['url']})")

    # export chat
    st.subheader("Export Conversation")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download as TXT"):
            buf = io.StringIO()
            for m in st.session_state.chat:
                buf.write(f"{m['role'].upper()}: {m['content']}\n\n")
            st.download_button("Save TXT", data=buf.getvalue(), file_name="sundarbot_chat.txt")
    with col2:
        if st.button("Generate PDF (optional)"):
            try:
                from reportlab.lib.pagesizes import LETTER
                from reportlab.pdfgen import canvas
                pdf_buf = io.BytesIO()
                c = canvas.Canvas(pdf_buf, pagesize=LETTER)
                width, height = LETTER
                y = height - 50
                c.setFont("Helvetica", 11)
                c.drawString(50, y, "SundarBot Conversation Export"); y -= 20
                for m in st.session_state.chat:
                    lines = (m['role'].upper() + ": " + m['content']).split("\n")
                    for line in lines:
                        if y < 60:
                            c.showPage(); y = height - 50; c.setFont("Helvetica", 11)
                        c.drawString(50, y, line[:95]); y -= 16
                    y -= 8
                c.save()
                st.download_button("Save PDF", data=pdf_buf.getvalue(), file_name="sundarbot_chat.pdf")
            except Exception:
                st.warning("ReportLab not installed; falling back to TXT.")

# =========================================================
# Tab 2: Map
# =========================================================
with tabs[1]:
    st.subheader("🗺️ Interactive Map: POIs + User Places")

    df_poi = pd.DataFrame(POIS)
    center = [df_poi["lat"].mean(), df_poi["lon"].mean()]
    fmap = Map(location=center, zoom_start=9)

    # POIs
    for _, r in df_poi.iterrows():
        Marker(
            location=[r.lat, r.lon],
            popup=Popup(f"<b>{r['name']}</b><br/>{r['type']} — {r['tips']}", max_width=260),
            icon=Icon(color="green" if r["type"]!="Visitor" else "blue", icon="info-sign"),
        ).add_to(fmap)

    # User + default places
    ALL_PLACES_DF = combined_places(LOCAL_PLACES)
    highlight_names = {r['name'] for r in st.session_state.get("last_recs", [])} if st.session_state.get("last_recs") else set()
    for _, r in ALL_PLACES_DF.iterrows():
        color = ("red" if r.get("cat") in ["hotel","restaurant"] else
                 "green" if r.get("cat") in ["vendor","shop"] else
                 "blue")
        if r["name"] in highlight_names: color = "red"
        popup_html = f"<b>{r['name']}</b><br/>{r.get('cat','')}<br/>{r.get('address','')}"
        if r.get("phone"):  popup_html += f"<br/>📞 {r['phone']}"
        if r.get("hours"):  popup_html += f"<br/>⏰ {r['hours']}"
        if r.get("website"):popup_html += f"<br/><a href='{r['website']}' target='_blank'>Website</a>"
        Marker(
            location=[float(r["lat"]), float(r["lon"])],
            popup=Popup(popup_html, max_width=280),
            icon=Icon(color=color, icon="info-sign"),
        ).add_to(fmap)

    # Fit bounds so there’s no empty gap
    bounds = [[r.lat, r.lon] for _, r in df_poi.iterrows()] + \
             [[float(r["lat"]), float(r["lon"])] for _, r in ALL_PLACES_DF.iterrows()]
    if bounds:
        fmap.fit_bounds(bounds)

    st_folium(fmap, height=520, use_container_width=True)

# =========================================================
# Tab 3: Submit (feedback & sightings)
# =========================================================
with tabs[2]:
    st.subheader("📝 Submit ideas or wildlife sightings")
    with st.form("feedback_form"):
        name = st.text_input("Your name (optional)")
        email = st.text_input("Email (optional)")
        entry_type = st.selectbox("Submission type", ["Idea","Bug","Wildlife Sighting"])
        details = st.text_area("Details (species, location, date/time, or idea)")
        agree = st.checkbox("I consent to store this info for research/feature improvements.")
        submitted = st.form_submit_button("Submit")
        if submitted:
            if not details:
                st.error("Please add some details.")
            elif not agree:
                st.warning("Please consent to store the submission.")
            else:
                ts = int(time.time())
                row = {"timestamp": ts, "name": name, "email": email, "type": entry_type, "details": details}
                try:
                    df_sub = pd.read_csv(SUBMISSIONS_CSV)
                except Exception:
                    df_sub = pd.DataFrame(columns=row.keys())
                df_sub = pd.concat([df_sub, pd.DataFrame([row])], ignore_index=True)
                df_sub.to_csv(SUBMISSIONS_CSV, index=False)
                st.success("Thanks! Your submission was recorded.")
                st.download_button("Download your submission (JSON)", data=json.dumps(row, indent=2), file_name=f"submission_{ts}.json")

# =========================================================
# Tab 4: Best Places (recommender UI)
# =========================================================
with tabs[3]:
    st.subheader("⭐ Best Places (by season & interests)")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        season = st.selectbox("Season", ["winter","post-monsoon","pre-monsoon","monsoon","summer"], index=0)
    with c2:
        time_of_day = st.selectbox("Time of day", ["morning","late-afternoon","afternoon","evening"], index=0)
    with c3:
        interests = st.multiselect("Interests", ["birding","family","beginner","education","canopy","crocodile","tiger","advanced"], default=["birding"])

    recs = recommend_pois(interests, season=season, time_of_day=time_of_day, k=5)
    st.session_state.last_recs = recs  # reuse in map
    for r in recs:
        st.markdown(f"**{r['name']}** — {r['type']} (score {r['score']})  \n*Why:* {r.get('tips','')}")
        st.caption(f"Best seasons: {', '.join(r.get('best_seasons', []))} • Best times: {', '.join(r.get('best_times', []))}")
        if r.get("image_url"): st.image(r["image_url"], use_container_width=True)
        st.divider()

    # Events / important dates
    st.subheader("🗓 Tourist spots & important dates")
    EVENTS = [
        {"title":"Peak winter birding","from":"2025-12-01","to":"2026-02-15","spot":"Multiple watchtowers","notes":"Migratory shorebirds & raptors"},
        {"title":"Monsoon caution","from":"2025-06-10","to":"2025-09-15","spot":"Creeks","notes":"Variable access; check permits & weather"},
        {"title":"Dobanki canopy walk maintenance","from":"2025-08-01","to":"2025-08-07","spot":"Dobanki","notes":"Expect brief closures"},
    ]
    def upcoming_events(n=6):
        today = date.today()
        rows = []
        for e in EVENTS:
            start = date.fromisoformat(e["from"])
            end   = date.fromisoformat(e["to"])
            if end >= today:
                rows.append((start, end, e))
        rows.sort(key=lambda x: x[0])
        return [e for _,__,e in rows][:n]

    with st.expander("View upcoming"):
        for e in upcoming_events():
            st.markdown(f"**{e['title']}** — {e['spot']}  \n{e['from']} → {e['to']}  \n*{e['notes']}*")

# =========================================================
# Tab 5: Weather
# =========================================================
with tabs[4]:
    st.subheader("🌧 Weather")
    st.info("Weather for the Sundarban center; you can wire GPS/user selection later.")
    render_weather_panel(*DEFAULT_CENTER)

# =========================================================
# Tab 6: Gallery (static URLs today)
# =========================================================
# ----------------------------
# Tab 6 (index 5): Image Gallery with Upload
# ----------------------------
# ----------------------------
# Tab 6 (index 5): Image Gallery with safe base files + Upload
# ----------------------------
# ============================
# FULL GALLERY TAB (drop-in)
# Tab index: 5  ->  with tabs[5]:
# Features:
#  - Shows base images (if present on disk)
#  - Anyone can upload JPG/PNG to assets/gallery/
#  - Persists uploads to gallery.csv
#  - Works on Windows/macOS/Linux (uses absolute paths)
#  - Compatible with Streamlit >= 1.31 (uses st.rerun fallback)
# ============================

with tabs[5]:
    st.subheader("🖼 Image Gallery")

    # ---------- helpers & setup ----------
    import os, re, time
    from pathlib import Path
    import pandas as pd

    # Safe rerun across Streamlit versions
    def _rerun():
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

    BASE_DIR = Path(__file__).parent.resolve()
    GALLERY_DIR = BASE_DIR / "assets" / "gallery"
    GALLERY_CSV = BASE_DIR / "gallery.csv"
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    # Base images — these are just placeholders. Keep/remove/edit as you like.
    # Files are optional: if not found, they will be skipped (no crash).
    BASE_GALLERY = [
        {"caption": "Creek reflections", "url": GALLERY_DIR / "creek.jpg"},
        {"caption": "Dobanki canopy",    "url": GALLERY_DIR / "dobanki.jpg"},
        {"caption": "Tour boats",        "url": GALLERY_DIR / "boats.jpg"},
        {"caption": "Boat at dusk",      "url": GALLERY_DIR / "dusk.jpg"},
    ]

    # CSV helpers for uploaded items
    def _to_records(df: pd.DataFrame) -> list[dict]:
        return df.to_dict("records") if not df.empty else []

    def load_uploaded() -> list[dict]:
        if GALLERY_CSV.exists():
            df = pd.read_csv(GALLERY_CSV)
            # ensure columns exist
            for c in ["caption", "url", "timestamp"]:
                if c not in df.columns:
                    df[c] = ""
            return _to_records(df[["caption", "url", "timestamp"]])
        return []

    def append_uploaded(row: dict):
        if GALLERY_CSV.exists():
            old = pd.read_csv(GALLERY_CSV)
            df = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(GALLERY_CSV, index=False)

    # Compose gallery items (base + uploaded)
    def _path_to_str(p: Path) -> str:
        return p.as_posix()

    items: list[dict] = []
    # include base files that actually exist
    for it in BASE_GALLERY:
        p: Path = it["url"]
        if isinstance(p, Path) and p.is_file():
            items.append({"caption": it["caption"], "url": _path_to_str(p)})

    # include uploaded items from CSV
    items.extend(load_uploaded())

    # ---------- renderer ----------
    def render_gallery(items: list[dict], cols: int = 3):
        if not items:
            st.info("No gallery images yet. Upload some below 👇")
            return

        rows = [items[i:i+cols] for i in range(0, len(items), cols)]
        for row in rows:
            cs = st.columns(len(row))
            for c, it in zip(cs, row):
                with c:
                    url = it.get("url", "")
                    cap = it.get("caption", "")
                    # local file
                    if url and os.path.isfile(url):
                        st.image(url, use_container_width=True, caption=cap)
                    # allow http(s) if you later add remote images
                    elif url.startswith("http://") or url.startswith("https://"):
                        st.image(url, use_container_width=True, caption=cap)
                    else:
                        st.warning(f"Missing file: {url}")

    render_gallery(items, cols=3)
    st.divider()

    # ---------- upload form ----------
    st.markdown("### ➕ Add a photo")
    with st.form("gallery_upload", clear_on_submit=True):
        file = st.file_uploader("Select image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        cap  = st.text_input("Caption")
        submit = st.form_submit_button("Upload")

        if submit:
            if not file:
                st.error("Please select an image.")
            else:
                # unique, safe filename under assets/gallery/
                base = re.sub(r"[^a-zA-Z0-9._-]", "_", file.name)
                fname = f"{int(time.time())}_{base}"
                save_path = GALLERY_DIR / fname
                with open(save_path, "wb") as f:
                    f.write(file.getbuffer())

                row = {
                    "caption": cap or Path(file.name).stem,
                    "url": _path_to_str(save_path),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                append_uploaded(row)

                st.success("Uploaded! Refreshing…")
                _rerun()

# =========================================================
# Tab 7: Add Places (Hotel + Shop/Vendor) with map picker
# =========================================================
with tabs[6]:
    st.subheader("➕ Add a Place (Hotel or Nearby Shop/Vendor)")
    st.caption("Pick a location by clicking the map (you can also type coordinates).")
    picker_center = [DEFAULT_CENTER[0], DEFAULT_CENTER[1]]
    picker_map = Map(location=picker_center, zoom_start=10)
    pick_state = st_folium(picker_map, height=320, use_container_width=True, key="place_picker")
    clicked = (pick_state.get("last_clicked") or {})
    clicked_lat = clicked.get("lat")
    clicked_lng = clicked.get("lng")

    colH, colS = st.columns(2)
    with colH:
        st.markdown("### 🏨 Add Hotel")
        with st.form("add_hotel"):
            h_name = st.text_input("Hotel name *")
            h_lat = st.number_input("Lat *", value=float(clicked_lat or DEFAULT_CENTER[0]), format="%.6f")
            h_lon = st.number_input("Lon *", value=float(clicked_lng or DEFAULT_CENTER[1]), format="%.6f")
            h_address = st.text_input("Address")
            h_phone = st.text_input("Phone / WhatsApp")
            h_hours = st.text_input("Hours", value="24h")
            h_site = st.text_input("Website / Booking URL")
            h_img = st.text_input("Image URL (optional)")
            h_notes = st.text_area("Notes (amenities, price range, tips)")
            ok_h = st.form_submit_button("Save Hotel")
            if ok_h:
                if not h_name:
                    st.error("Hotel name is required")
                else:
                    save_user_place({
                        "name": h_name, "cat": "hotel",
                        "lat": h_lat, "lon": h_lon,
                        "address": h_address, "phone": h_phone, "hours": h_hours,
                        "website": h_site, "notes": h_notes, "image_url": h_img,
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds")+"Z",
                    })
                    st.success(f"Saved: {h_name}")

    with colS:
        st.markdown("### 🛒 Add Shop / Local Vendor")
        with st.form("add_shop"):
            s_name = st.text_input("Name *", key="shop_name")
            s_cat = st.selectbox("Category *", ["shop","vendor","permit","restaurant","medical"], index=0)
            s_lat = st.number_input("Lat *", value=float(clicked_lat or DEFAULT_CENTER[0]), format="%.6f", key="shop_lat")
            s_lon = st.number_input("Lon *", value=float(clicked_lng or DEFAULT_CENTER[1]), format="%.6f", key="shop_lon")
            s_address = st.text_input("Address", key="shop_addr")
            s_phone = st.text_input("Phone", key="shop_phone")
            s_hours = st.text_input("Hours", value="7:00–18:00", key="shop_hours")
            s_site = st.text_input("Website / Link", key="shop_site")
            s_img = st.text_input("Image URL (optional)", key="shop_img")
            s_notes = st.text_area("What they offer / tips", key="shop_notes")
            ok_s = st.form_submit_button("Save Entry")
            if ok_s:
                if not s_name:
                    st.error("Name is required")
                else:
                    save_user_place({
                        "name": s_name, "cat": s_cat,
                        "lat": s_lat, "lon": s_lon,
                        "address": s_address, "phone": s_phone, "hours": s_hours,
                        "website": s_site, "notes": s_notes, "image_url": s_img,
                        "timestamp": datetime.utcnow().isoformat(timespec="seconds")+"Z",
                    })
                    st.success(f"Saved: {s_name}")

    st.divider()
    st.caption("All added places are stored in `places.csv` next to the app.")
    st.dataframe(load_user_places())

# =========================================================
# Tab 8: Tour Companies (list-only)
# =========================================================
with tabs[7]:
    st.subheader("🏝️ Sundarban Tour Companies — Public List")
    st.caption("Anyone can add a tour operator here. Entries are stored in `tour_companies.csv`.")

    # Add form
    with st.form("add_tour_company"):
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Company name *")
            contact_person = st.text_input("Contact person")
            phone = st.text_input("Phone / WhatsApp *")
            email = st.text_input("Email")
            website = st.text_input("Website")
        with col2:
            address = st.text_area("Address / Office")
            license_no = st.text_input("License / Permit No.")
            services = st.text_area("Services (e.g., boat safari, permits, homestay, pickup)")
            notes = st.text_area("Notes (pricing, languages, tips)")
        ok = st.form_submit_button("Save Company")
        if ok:
            if not company_name or not phone:
                st.error("Company name and phone are required.")
            else:
                save_tour({
                    "company_name": company_name.strip(),
                    "contact_person": contact_person.strip(),
                    "phone": phone.strip(),
                    "email": email.strip(),
                    "website": website.strip(),
                    "address": address.strip(),
                    "license_no": license_no.strip(),
                    "services": services.strip(),
                    "notes": notes.strip(),
                    "timestamp": datetime.utcnow().isoformat(timespec="seconds")+"Z",
                })
                st.success(f"Saved: {company_name}")

    st.divider()

    # Search + table
    q = st.text_input("Quick search (name, phone, services, notes)")
    df_tour = load_tours()
    if q:
        ql = q.lower()
        mask = (
            df_tour.fillna("").apply(lambda row:
                ql in (" ".join([str(x) for x in row.values])).lower(), axis=1
            )
        )
        df_tour = df_tour[mask]

    st.dataframe(df_tour, use_container_width=True)

    # Export
    if not df_tour.empty:
        st.download_button("Download CSV", data=df_tour.to_csv(index=False), file_name="tour_companies.csv")

# =========================================================
# Tab 9: Knowledge Base
# =========================================================
with tabs[8]:
    st.subheader("📚 Knowledge Base")
    sect = st.multiselect("Filter by section", sorted({d["section"] for d in KB_DOCS}), default=[])
    q = st.text_input("Search KB")

    filtered = KB_DOCS
    if sect: filtered = [d for d in filtered if d["section"] in sect]
    if q:    filtered = [d for d in filtered if q.lower() in (d["title"]+" "+d["text"]).lower()]

    for d in filtered:
        with st.expander(f"{d['title']} — {d['section']}"):
            st.write(d["text"])
            if d.get("refs"):
                st.caption("References:")
                for r in d["refs"]:
                    st.markdown(f"- [{r['label']}]({r['url']})")

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption("© 2025 SundarBot • Built with Streamlit • Colors: green/teal/sandy • Icon: mangrove + circuit motif")
