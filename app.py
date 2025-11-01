import io, re, base64
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st
from google_play_scraper import app as gp_app, reviews, Sort
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------- Page config: biarkan sidebar default (desktop aman)
st.set_page_config(
    page_title="UlasAnalisa – Prediksi Sentimen",
    page_icon="static/logo_ulas.png",
    layout="wide",
    initial_sidebar_state="expanded",   # desktop: sidebar tampil normal, tidak ketumpuk
)

# ---------------- State
for k, v in {
    "results": {}, "app_id": None, "csv_pred": None, "csv_dist": None, "is_combo": False
}.items():
    if k not in st.session_state: st.session_state[k] = v

# ---------------- Global CSS (tidak menyentuh sidebar untuk desktop)
st.markdown("""
<style>
:root{ --nav-h: 80px; } /* samakan dengan tinggi navbar di bawah */

/* header jangan dihilangkan - hamburger butuh ini */
[data-testid="stHeader"]{
  background: transparent !important; box-shadow:none !important; min-height:0 !important; height:auto !important;
}

/* offset konten karena navbar fixed */
[data-testid="stAppViewContainer"] > .main{ margin-top: var(--nav-h) !important; }

/* navbar */
.navbar{ z-index: 1000 !important; }

/* MOBILE ONLY: pindahkan hamburger di bawah navbar. Tidak ubah layout desktop. */
@media (max-width: 900px){
  [data-testid="stSidebarCollapseButton"]{
    position: fixed !important; top: var(--nav-h) !important; left: 10px !important;
    z-index: 1002 !important; display: flex !important;
  }
  /* Sidebar overlay tetap milik Streamlit; hanya digeser ke bawah navbar */
  [data-testid="stSidebar"]{
    top: var(--nav-h) !important; height: calc(100% - var(--nav-h)) !important; z-index: 1001 !important;
  }
}
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers
def img_to_base64(path: str) -> str:
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode()

# ---------------- Navbar (custom ringan)
logo_left_b64  = img_to_base64("static/logo_ulas.png")
logo_right_b64 = img_to_base64("static/fti_untar.png")
try:
    page = st.query_params.get("page", "home")
except AttributeError:
    page = st.experimental_get_query_params().get("page", ["home"])[0]
home_a  = "active" if page=="home" else ""
pred_a  = "active" if page=="prediksi" else ""
tent_a  = "active" if page=="tentang" else ""

st.markdown(f"""
<style>
.navbar {{
  position: fixed; top:0; left:0; right:0; height:80px; background:#fff;
  display:flex; align-items:center; padding:0 1.5rem; border-bottom:3px solid #b71c1c;
}}
.nav-left,.nav-right {{ width:220px; display:flex; justify-content:center; align-items:center; }}
.nav-center {{ flex:1; display:flex; justify-content:center; gap:2.5rem; }}
.nav-center a {{ text-decoration:none; color:#444; font-weight:500; }}
.nav-center a.active {{ color:#b71c1c; border-bottom:2px solid #b71c1c; padding-bottom:4px; }}
.logo-left{{ height:150px; }} .logo-right{{ height:65px; }}
</style>
<div class="navbar">
  <div class="nav-left"><img src="data:image/png;base64,{logo_left_b64}" class="logo-left"></div>
  <div class="nav-center">
    <a href="?page=home" target="_self" class="{home_a}">Beranda</a>
    <a href="?page=prediksi" target="_self" class="{pred_a}">Prediksi</a>
    <a href="?page=tentang" target="_self" class="{tent_a}">Tentang</a>
  </div>
  <div class="nav-right"><img src="data:image/png;base64,{logo_right_b64}" class="logo-right"></div>
</div>
""", unsafe_allow_html=True)

# ---------------- HOME
if page == "home":
    st.markdown("## Selamat datang di **UlasAnalisa**")
    st.markdown("### Apa itu **UlasAnalisa?**")
    st.write("Analisis sentimen ulasan Google Play otomatis. Hasil bisa diunduh **CSV**.")
    st.markdown("---"); st.write("### Cara pakai singkat")
    c1,c2 = st.columns([1,1])
    with c1: st.image("static/1.png", use_container_width=True)
    with c2: st.write("Step 1: Copy URL aplikasi dari Google Play (web).")
    st.markdown("---")
    st.write("Step 1 (HP): Play Store → ⋮ → Share → Copy URL.")
    a,b,c = st.columns(3)
    with a: st.image("static/2.png", use_container_width=True)
    with b: st.image("static/3.png", use_container_width=True)
    with c: st.image("static/4.png", use_container_width=True)
    st.markdown("---")
    c1,c2 = st.columns([1,1])
    with c1: st.image("static/5.png", use_container_width=True)
    with c2: st.write("Step 2: Paste URL di halaman **Prediksi**.")
    st.markdown("---")
    c1,c2 = st.columns([1,1])
    with c1: st.image("static/6.png", use_container_width=True)
    with c2: st.write("Step 3: Atur model, bahasa, negara, jumlah ulasan, urutan → klik **Prediksi**.")

# ---------------- TENTANG
elif page == "tentang":
    st.markdown("### Pengembang Website")
    c1,c2 = st.columns([1,2])
    with c1: st.image("static/fotoku.png", width=180)
    with c2:
        st.markdown("""
**Nama** : Parveen Uzma Habidin  
**NIM** : 535220226  
**Jurusan** : Teknik Informatika  
**Fakultas** : Teknik Informasi  

**Topik Skripsi** :  
Perencanaan Analisis Sentimen Aplikasi Sosial Media Pada Google Play Store Menggunakan Model Random Forest, Support Vector Machine dan TF-IDF
""")
    st.markdown("---")
    a,b = st.columns(2)
    with a:
        st.markdown("### Dosen Pembimbing"); st.image("static/pak_tri.webp", width=140)
        st.markdown("Tri Sutrisno, S.Si., M.Sc.")
    with b:
        st.markdown("### Institusi"); st.image("static/Logo_untar.png", width=180)
        st.markdown("**Universitas Tarumanagara**")

# ---------------- PREDIKSI
elif page == "prediksi":
    
    # --- Paksa sidebar terbuka di desktop (kalau tersimpan collapsed di local storage) ---
    st.markdown("""
    <script>
    (function() {
    function openSidebarIfCollapsed() {
        try {
        const d = window.parent.document;
        const btn = d.querySelector('[data-testid="stSidebarCollapseButton"] button');
        const sb  = d.querySelector('[data-testid="stSidebar"]');
        if (!btn || !sb) return;
        const isCollapsed = sb.getAttribute('aria-expanded') === 'false';
        const isDesktop   = window.innerWidth >= 901;
        if (isDesktop && isCollapsed) btn.click();
        } catch (e) { /* ignore */ }
    }
    setTimeout(openSidebarIfCollapsed, 150);
    setTimeout(openSidebarIfCollapsed, 400);
    setTimeout(openSidebarIfCollapsed, 800);
    window.addEventListener('resize', openSidebarIfCollapsed);
    })();
    </script>
    """, unsafe_allow_html=True)

    
    st.title("Prediksi Sentimen dari Link Google Play")
    st.caption("Masukkan link aplikasi dari Google Play Store, lalu sistem akan prediksi sentimennya")

    VEC_PATH = Path("Artifacts") / "tfidf_vectorizer.joblib"
    SVM_PATH = Path("Artifacts") / "svm_rbf_model.joblib"
    RF_PATH  = Path("Artifacts") / "random_forest_model.joblib"

    @st.cache_resource
    def load_artifacts():
        vec = joblib.load(VEC_PATH)
        svm = joblib.load(SVM_PATH) if SVM_PATH.exists() else None
        rf  = joblib.load(RF_PATH)  if RF_PATH.exists()  else None
        return vec, svm, rf

    try:
        tfidf_vectorizer, svm_model, rf_model = load_artifacts()
    except Exception as e:
        st.error(f"Gagal memuat artifacts.\nDetail: {e}"); st.stop()

    avail = []
    if svm_model is not None: avail.append("SVM (RBF)")
    if rf_model  is not None: avail.append("RandomForest")
    if svm_model is not None and rf_model is not None: avail.append("SVM dan RandomForest")
    if not avail: st.error("Tidak ada model yang tersedia."); st.stop()

    with st.sidebar:
        st.header("Pengaturan")
        model_name = st.selectbox("Pilih model", avail, index=0)
        lang       = st.selectbox("Bahasa ulasan", ["id","en"], index=0)
        country    = st.selectbox("Negara", ["id","us"], index=0)
        n_reviews  = st.slider("Jumlah ulasan di-scrape", 50, 1000, 200, 50)
        sort_opt   = st.selectbox("Urutkan", ["NEWEST","MOST_RELEVANT"], index=0)
        run        = st.button("Prediksi")

    ID_RE = re.compile(r"[?&]id=([a-zA-Z0-9._]+)")
    def parse_app_id(text: str) -> str:
        t = (text or "").strip(); m = ID_RE.search(t); return m.group(1) if m else t

    def scrape_reviews(app_id: str, lang="id", country="id", n=200, sort="NEWEST"):
        sort_map = {"NEWEST": Sort.NEWEST, "MOST_RELEVANT": Sort.MOST_RELEVANT}
        sort = sort_map.get(sort, Sort.NEWEST)
        got, token = [], None
        while len(got) < n:
            batch, token = reviews(app_id, lang=lang, country=country, sort=sort,
                                   count=min(200, n-len(got)), continuation_token=token)
            got.extend(batch)
            if token is None: break
        if not got: return pd.DataFrame(columns=["content","score","at","replyContent","userName"])
        return pd.DataFrame(got)

    link = st.text_input("Masukkan link Google Play / package id",
                         placeholder="https://play.google.com/store/apps/details?id=com.zhiliaoapp.musically")

    if run:
        app_id = parse_app_id(link)
        if not app_id: st.error("Package id tidak valid."); st.stop()

        try:
            meta = gp_app(app_id, lang=lang, country=country)
            st.markdown(f"**App:** {meta.get('title','?')}  \n**Package:** `{app_id}`  \n**Installs:** {meta.get('installs','?')}  \n**Score:** {meta.get('score','?')}")
        except Exception:
            st.info(f"Package: `{app_id}`")

        with st.spinner(f"Mengambil {n_reviews} ulasan..."):
            df = scrape_reviews(app_id, lang=lang, country=country, n=n_reviews, sort=sort_opt)
        if df.empty: st.warning("Tidak ada ulasan yang diambil."); st.stop()

        df = df.rename(columns={"content":"text","score":"rating","at":"date"})
        cols = ["text","rating","date","userName","replyContent"]
        df  = df[[c for c in cols if c in df.columns]].copy()

        with st.spinner("Mengubah fitur (TF-IDF) dan memprediksi"):
            X = joblib.load(VEC_PATH).transform(df["text"].astype(str)).toarray()
            results = {}
            if model_name == "SVM (RBF)":
                y = joblib.load(SVM_PATH).predict(X)
                t = df.copy(); t["pred"]=y; t["pred_label"]=t["pred"].map({1:"Positive",0:"Negative"})
                results["SVM (RBF)"]=t
            elif model_name == "RandomForest":
                y = joblib.load(RF_PATH).predict(X)
                t = df.copy(); t["pred"]=y; t["pred_label"]=t["pred"].map({1:"Positive",0:"Negative"})
                results["RandomForest"]=t
            else:
                y_svm = joblib.load(SVM_PATH).predict(X)
                t1=df.copy(); t1["pred"]=y_svm; t1["pred_label"]=t1["pred"].map({1:"Positive",0:"Negative"})
                results["SVM (RBF)"]=t1
                y_rf  = joblib.load(RF_PATH).predict(X)
                t2=df.copy(); t2["pred"]=y_rf; t2["pred_label"]=t2["pred"].map({1:"Positive",0:"Negative"})
                results["RandomForest"]=t2

        st.session_state.results = results
        st.session_state.app_id  = app_id
        st.session_state.is_combo = (model_name=="SVM dan RandomForest")

        if st.session_state.is_combo:
            df_svm = results["SVM (RBF)"]; df_rf = results["RandomForest"]
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as w:
                df_svm.to_excel(w, sheet_name="SVM (RBF)", index=False)
                df_rf.to_excel(w,  sheet_name="RandomForest", index=False)
            out.seek(0); st.session_state.csv_pred = out.getvalue()

            dist_svm = df_svm["pred_label"].value_counts().rename_axis("sentiment").reset_index(name="count")
            dist_rf  = df_rf["pred_label"].value_counts().rename_axis("sentiment").reset_index(name="count")
            out2 = io.BytesIO()
            with pd.ExcelWriter(out2, engine="xlsxwriter") as w:
                dist_svm.to_excel(w, sheet_name="SVM (RBF)", index=False)
                dist_rf.to_excel(w,  sheet_name="RandomForest", index=False)
            out2.seek(0); st.session_state.csv_dist = out2.getvalue()
        else:
            key = next(iter(results)); dfk = results[key]
            st.session_state.csv_pred = dfk.to_csv(index=False).encode("utf-8")
            st.session_state.csv_dist = dfk["pred_label"].value_counts().rename_axis("sentiment").reset_index(name="count").to_csv(index=False).encode("utf-8")

# ---------------- OUTPUT
if st.session_state.results and page=="prediksi":
    items = list(st.session_state.results.items())
    cols = st.columns(len(items))
    for c,(name,dfm) in zip(cols, items):
        with c:
            st.subheader(f"Distribusi Sentimen – {name}")
            dist = dfm["pred_label"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
            st.bar_chart(dist.set_index("Sentiment"))
            st.subheader(f"Sampel Hasil Prediksi – {name}")
            st.dataframe(dfm.head(20), use_container_width=True)

    c1,c2,c3,c4,c5 = st.columns([1,2,2,2,1])
    with c2:
        st.download_button(
            "Download Hasil Prediksi", data=st.session_state.csv_pred,
            file_name=(f"{st.session_state.app_id}_prediksi_ulasan.xlsx" if st.session_state.is_combo else f"{st.session_state.app_id}_prediksi_ulasan.csv"),
            mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if st.session_state.is_combo else "text/csv"),
            type="primary", key="dl_pred"
        )
    with c4:
        st.download_button(
            "Download Distribusi Sentimen", data=st.session_state.csv_dist,
            file_name=(f"{st.session_state.app_id}_distribusi_sentimen.xlsx" if st.session_state.is_combo else f"{st.session_state.app_id}_distribusi_sentimen.csv"),
            mime=("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if st.session_state.is_combo else "text/csv"),
            type="primary", key="dl_dist"
        )

    def rating_to_label(r): 
        if pd.isna(r): return None
        return 1 if r >= 4 else 0
    metrics=[]
    for name,dfm in st.session_state.results.items():
        d= dfm.copy(); d["true_label"]=d["rating"].apply(rating_to_label); d=d.dropna(subset=["true_label"])
        if d.empty: continue
        metrics.append({
            "Model":name,
            "Accuracy":accuracy_score(d["true_label"], d["pred"]),
            "Precision":precision_score(d["true_label"], d["pred"]),
            "Recall":recall_score(d["true_label"], d["pred"]),
            "F1-Score":f1_score(d["true_label"], d["pred"]),
        })
    if metrics: st.subheader("Perbandingan Metrik Evaluasi (dibanding rating bintang)"); st.dataframe(pd.DataFrame(metrics), use_container_width=True)
    else: st.info("Tidak ada metrik yang bisa dihitung.")
elif page=="prediksi" and not st.session_state.results:
    st.info("Masukkan link/package, lalu klik **Prediksi**.")
