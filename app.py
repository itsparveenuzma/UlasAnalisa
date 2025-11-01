import io
import re
import base64
from pathlib import Path

import pandas as pd
import joblib
import streamlit as st
from google_play_scraper import app as gp_app, reviews, Sort
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="UlasAnalisa – Prediksi Sentimen",
    page_icon="static/logo_ulas.png",
    layout="wide",
    initial_sidebar_state="collapsed",   # mobile butuh hamburger; desktop kita paksa tampil via CSS
)

# =========================
# SESSION STATE BOOTSTRAP
# =========================
defaults = {
    "results": {},          # dict: {model_name: df}
    "app_id": None,
    "csv_pred": None,
    "csv_dist": None,
    "is_combo": False,      # unduhan xlsx saat dua model
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================
# GLOBAL (LIGHT) CSS
# - header jangan disembunyikan agar hamburger bisa muncul di mobile
# - beri offset konten karena navbar fixed
# =========================
st.markdown("""
<style>
:root{
  --nav-h: 90px;
}

/* header Streamlit tetap ada (tipis) */
[data-testid="stHeader"]{
  background: transparent !important;
  box-shadow: none !important;
  min-height: 0 !important;
  height: auto !important;
}

/* offset konten global karena navbar custom fixed */
[data-testid="stAppViewContainer"] > .main{
  margin-top: var(--nav-h) !important;
}

/* navbar selalu di atas */
.navbar{ z-index: 100000 !important; }
</style>
""", unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# =========================
# NAVBAR (custom)
# =========================
logo_left_b64 = img_to_base64("static/logo_ulas.png")
logo_right_b64 = img_to_base64("static/fti_untar.png")

# baca query ?page=
try:  # Streamlit baru
    page = st.query_params.get("page", "home")
except AttributeError:  # fallback Streamlit lama
    page = st.experimental_get_query_params().get("page", ["home"])[0]

home_active    = "active" if page == "home"     else ""
pred_active    = "active" if page == "prediksi" else ""
tentang_active = "active" if page == "tentang"  else ""

st.markdown(f"""
<style>
.navbar {{
  position: fixed; top: 0; left: 0; right: 0;
  height: 80px; background: #ffffff;
  display: flex; align-items: center;
  padding: 0 1.5rem;
  border-bottom: 3px solid #b71c1c;
}}

[data-testid="stAppViewContainer"] > .main {{ margin-top: var(--nav-h); }}

.nav-left, .nav-right {{
  width: 220px; display: flex; justify-content: center; align-items: center;
}}
.nav-center {{
  flex: 1; display: flex; justify-content: center; gap: 2.5rem;
}}
.nav-center a {{
  text-decoration: none; color: #444; font-weight: 500;
}}
.nav-center a.active {{
  color: #b71c1c; border-bottom: 2px solid #b71c1c; padding-bottom: 4px;
}}
.logo-left  {{ height: 150px; }}
.logo-right {{ height: 65px; }}
</style>

<div class="navbar">
  <div class="nav-left">
    <img src="data:image/png;base64,{logo_left_b64}" class="logo-left">
  </div>
  <div class="nav-center">
    <a href="?page=home" target="_self" class="{home_active}">Beranda</a>
    <a href="?page=prediksi" target="_self" class="{pred_active}">Prediksi</a>
    <a href="?page=tentang" target="_self" class="{tentang_active}">Tentang</a>
  </div>
  <div class="nav-right">
    <img src="data:image/png;base64,{logo_right_b64}" class="logo-right">
  </div>
</div>
""", unsafe_allow_html=True)


# =========================
# LAYOUT CSS PER-HALAMAN
# =========================
if page == "prediksi":
    # DESKTOP: sidebar tetap, konten geser & center
    # MOBILE : hamburger di bawah navbar (kiri), sidebar overlay
    st.markdown("""
    <style>
      :root{
        --nav-h: 90px;
        --sb-w: 19rem;
        --content-max: 1100px;
      }

      /* ===== DESKTOP (>=901px): sidebar permanen di kiri ===== */
      @media (min-width: 901px){
        [data-testid="stSidebar"]{
          position: fixed !important;
          top: var(--nav-h) !important;
          left: 0 !important;
          width: var(--sb-w) !important;
          height: calc(100% - var(--nav-h)) !important;
          overflow-y: auto !important;
          background: var(--color-bg, #111) !important;
          border-right: 1px solid rgba(255,255,255,.08) !important;
          z-index: 9999 !important;
          transform: none !important;   /* paksa tampil walau state collapsed */
          box-shadow: none !important;
        }

        /* konten geser sejauh lebar sidebar */
        [data-testid="stAppViewContainer"] > .main{
          margin-top: var(--nav-h) !important;
          margin-left: var(--sb-w) !important;
        }

        [data-testid="stAppViewContainer"] > .main .block-container{
          max-width: var(--content-max) !important;
          margin-left: auto !important;
          margin-right: auto !important;
          padding-left: 1.25rem !important;
          padding-right: 1.25rem !important;
        }

        /* sembunyikan tombol collapse di desktop */
        [data-testid="stSidebarCollapseButton"]{ display: none !important; }
      }

      /* ===== MOBILE (<=900px): hamburger di bawah navbar, sidebar overlay ===== */
      @media (max-width: 900px){
        /* tombol hamburger di bawah navbar kiri */
        [data-testid="stSidebarCollapseButton"]{
          position: fixed !important;
          top: var(--nav-h) !important;   /* tepat di bawah navbar */
          left: 10px !important;
          z-index: 200001 !important;
          display: flex !important;
        }

        /* sidebar sebagai drawer/overlay */
        [data-testid="stSidebar"]{
          position: fixed !important;
          top: var(--nav-h) !important;
          left: 0 !important;
          width: 80vw !important;
          max-width: 22rem !important;
          height: calc(100% - var(--nav-h)) !important;
          overflow-y: auto !important;
          background: var(--color-bg, #111) !important;
          z-index: 200000 !important;

          transform: translateX(-100%) !important;
          transition: transform .25s ease-in-out !important;
          box-shadow: none !important;
        }
        [data-testid="stSidebar"][aria-expanded="true"]{
          transform: translateX(0) !important;
        }

        /* konten full width */
        [data-testid="stAppViewContainer"] > .main{
          margin-top: var(--nav-h) !important;
          margin-left: 0 !important;
        }
      }
    </style>
    """, unsafe_allow_html=True)

else:
    # HOME/TENTANG: tanpa sidebar, center content
    st.markdown("""
    <style>
      :root{ --nav-h: 90px; --content-max: 1100px; }

      [data-testid="stAppViewContainer"] > .main{
        margin-top: var(--nav-h) !important;
        margin-left: 0 !important;
      }
      [data-testid="stAppViewContainer"] > .main .block-container{
        max-width: var(--content-max) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
      }

      /* pastikan tombol collapse tidak tampil di halaman tanpa sidebar */
      [data-testid="stSidebarCollapseButton"]{ display: none !important; }
    </style>
    """, unsafe_allow_html=True)


# =========================
# HALAMAN: HOME
# =========================
if page == "home":
    st.markdown("## Selamat datang di **UlasAnalisa**")
    st.markdown("### Apa itu **UlasAnalisa?**")
    st.markdown(
        """
        **UlasAnalisa** adalah website yang membantu menganalisis sentimen ulasan aplikasi di Google Play Store secara otomatis dan menyajikannya dalam bentuk tabel yang mudah dipahami.  
        Hasil sentimen bisa diunduh dalam bentuk **.csv**.
        """
    )

    st.markdown("### Bagaimana Cara Memakainya?")

    # STEP 1 (Website)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("static/1.png", caption="Tampilan Google Play di Website", use_container_width=True)
    with col2:
        st.markdown("### Step 1 (Website)")
        st.write("Copy link aplikasi dari halaman Google Play Store yang ingin dianalisa (website).")

    st.markdown("---")

    # STEP 1 (Handphone)
    st.markdown("### Step 1 (Handphone)")
    sp1, c1, c2, c3, sp2 = st.columns([1, 2, 2, 2, 1])
    with c1:
        st.image("static/2.png", width=230)
    with c2:
        st.image("static/3.png", width=230)
    with c3:
        st.image("static/4.png", width=230)
    st.write("Buka Google Play Store di HP → cari aplikasinya → ketuk **⋮ → Share** → pilih **Copy URL**.")

    st.markdown("---")

    # STEP 2
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("static/5.png", use_container_width=True)
    with col2:
        st.markdown("### Step 2")
        st.write("Paste / tempel link URL tadi ke kolom input di halaman **Prediksi**.")

    st.markdown("---")

    # STEP 3
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("static/6.png", use_container_width=True)
    with col2:
        st.markdown("### Step 3")
        st.write("Atur pengaturan (model, bahasa, negara, jumlah ulasan, urutan) sesuai kebutuhan.")

    st.markdown("---")

    # STEP 4
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("static/7.png", use_container_width=True)
    with col2:
        st.markdown("### Step 4")
        st.write("Klik tombol **Prediksi** → sistem akan ambil ulasan dan menampilkan hasil serta tombol download CSV.")


# =========================
# HALAMAN: TENTANG
# =========================
elif page == "tentang":
    st.markdown("### Pengembang Website")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("static/fotoku.png", width=180)
    with col2:
        st.markdown(
            """
            **Nama** : Parveen Uzma Habidin  
            **NIM** : 535220226  
            **Jurusan** : Teknik Informatika  
            **Fakultas** : Teknik Informasi  

            **Topik Skripsi** :  
            Perencanaan Analisis Sentimen Aplikasi Sosial Media Pada Google Play Store Menggunakan Model Random Forest, Support Vector Machine dan TF-IDF
            """
        )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Dosen Pembimbing")
        st.image("static/pak_tri.webp", width=140)
        st.markdown("Tri Sutrisno, S.Si., M.Sc.")
    with col2:
        st.markdown("### Institusi")
        st.image("static/Logo_untar.png", width=180)
        st.markdown("**Universitas Tarumanagara**")


# =========================
# HALAMAN: PREDIKSI
# =========================
elif page == "prediksi":

    st.title("Prediksi Sentimen dari Link Google Play")
    st.caption("Masukkan link aplikasi dari Google Play Store, lalu sistem akan prediksi sentimennya")

    # Artifacts path
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
        st.error(f"Gagal memuat artifacts.\nDetail: {e}")
        st.stop()

    # pilihan model
    avail = []
    if svm_model is not None: avail.append("SVM (RBF)")
    if rf_model  is not None: avail.append("RandomForest")
    if svm_model is not None and rf_model is not None:
        avail.append("SVM dan RandomForest")
    if not avail:
        st.error("Tidak ada model yang tersedia.")
        st.stop()

    # Sidebar form
    with st.sidebar:
        st.header("Pengaturan")
        model_name = st.selectbox("Pilih model", avail, index=0)
        lang = st.selectbox("Bahasa ulasan", ["id", "en"], index=0)
        country = st.selectbox("Negara", ["id", "us"], index=0)
        n_reviews = st.slider("Jumlah ulasan di-scrape", 50, 1000, 200, 50)
        sort_opt = st.selectbox("Urutkan", ["NEWEST", "MOST_RELEVANT"], index=0)
        run = st.button("Prediksi")

    # utils
    ID_RE = re.compile(r"[?&]id=([a-zA-Z0-9._]+)")
    def parse_app_id(text: str) -> str:
        t = (text or "").strip()
        m = ID_RE.search(t)
        return m.group(1) if m else t

    def scrape_reviews(app_id: str, lang="id", country="id", n=200, sort="NEWEST"):
        sort_map = {"NEWEST": Sort.NEWEST, "MOST_RELEVANT": Sort.MOST_RELEVANT}
        sort = sort_map.get(sort, Sort.NEWEST)
        got, token = [], None
        while len(got) < n:
            batch, token = reviews(
                app_id, lang=lang, country=country, sort=sort,
                count=min(200, n - len(got)), continuation_token=token
            )
            got.extend(batch)
            if token is None:
                break
        if not got:
            return pd.DataFrame(columns=["content","score","at","replyContent","userName"])
        return pd.DataFrame(got)

    link = st.text_input(
        "Masukkan link Google Play / package id",
        placeholder="https://play.google.com/store/apps/details?id=com.zhiliaoapp.musically"
    )

    if run:
        app_id = parse_app_id(link)
        if not app_id:
            st.error("Package id tidak valid.")
            st.stop()

        # tampilkan meta app
        try:
            meta = gp_app(app_id, lang=lang, country=country)
            st.markdown(
                f"**App:** {meta.get('title','?')}  \n"
                f"**Package:** `{app_id}`  \n"
                f"**Installs:** {meta.get('installs','?')}  \n"
                f"**Score:** {meta.get('score','?')}"
            )
        except Exception:
            st.info(f"Package: `{app_id}`")

        # ambil ulasan
        with st.spinner(f"Mengambil {n_reviews} ulasan..."):
            df = scrape_reviews(app_id, lang=lang, country=country, n=n_reviews, sort=sort_opt)
        if df.empty:
            st.warning("Tidak ada ulasan yang diambil.")
            st.stop()

        df = df.rename(columns={"content": "text", "score": "rating", "at": "date"})
        cols = ["text","rating","date","userName","replyContent"]
        df = df[[c for c in cols if c in df.columns]].copy()

        # transform & prediksi
        with st.spinner("Mengubah fitur (TF-IDF) dan memprediksi"):
            X_tfidf = tfidf_vectorizer.transform(df["text"].astype(str))
            X_dense = X_tfidf.toarray()

            results = {}

            if model_name == "SVM (RBF)":
                y_pred = svm_model.predict(X_dense)
                tmp = df.copy()
                tmp["pred"] = y_pred
                tmp["pred_label"] = tmp["pred"].map({1: "Positive", 0: "Negative"})
                results["SVM (RBF)"] = tmp

            elif model_name == "RandomForest":
                y_pred = rf_model.predict(X_dense)
                tmp = df.copy()
                tmp["pred"] = y_pred
                tmp["pred_label"] = tmp["pred"].map({1: "Positive", 0: "Negative"})
                results["RandomForest"] = tmp

            else:  # SVM dan RandomForest
                y_svm = svm_model.predict(X_dense)
                df_svm = df.copy()
                df_svm["pred"] = y_svm
                df_svm["pred_label"] = df_svm["pred"].map({1: "Positive", 0: "Negative"})
                results["SVM (RBF)"] = df_svm

                y_rf = rf_model.predict(X_dense)
                df_rf = df.copy()
                df_rf["pred"] = y_rf
                df_rf["pred_label"] = df_rf["pred"].map({1: "Positive", 0: "Negative"})
                results["RandomForest"] = df_rf

        # simpan ke session
        st.session_state.results = results
        st.session_state.app_id = app_id
        st.session_state.is_combo = (model_name == "SVM dan RandomForest")

        # siapkan file unduhan
        if st.session_state.is_combo:
            df_svm = results["SVM (RBF)"]
            df_rf  = results["RandomForest"]

            # workbook hasil
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_svm.to_excel(writer, sheet_name="SVM (RBF)", index=False)
                df_rf.to_excel(writer, sheet_name="RandomForest", index=False)
            output.seek(0)
            st.session_state.csv_pred = output.getvalue()

            # workbook distribusi
            dist_svm = df_svm["pred_label"].value_counts().rename_axis("sentiment").reset_index(name="count")
            dist_rf  = df_rf["pred_label"].value_counts().rename_axis("sentiment").reset_index(name="count")

            dist_out = io.BytesIO()
            with pd.ExcelWriter(dist_out, engine="xlsxwriter") as writer:
                dist_svm.to_excel(writer, sheet_name="SVM (RBF)", index=False)
                dist_rf.to_excel(writer, sheet_name="RandomForest", index=False)
            dist_out.seek(0)
            st.session_state.csv_dist = dist_out.getvalue()
        else:
            first_key = next(iter(results))
            first_df = results[first_key]
            st.session_state.csv_pred = first_df.to_csv(index=False).encode("utf-8")
            dist_df = first_df["pred_label"].value_counts().rename_axis("sentiment").reset_index(name="count")
            st.session_state.csv_dist = dist_df.to_csv(index=False).encode("utf-8")


# =========================
# OUTPUT HASIL (grafik, tabel, unduh, metrik)
# =========================
if st.session_state.results and page == "prediksi":
    models_items = list(st.session_state.results.items())

    cols = st.columns(len(models_items))
    for col, (model_key, df_model) in zip(cols, models_items):
        with col:
            st.subheader(f"Distribusi Sentimen – {model_key}")
            dist_df = (
                df_model["pred_label"]
                .value_counts()
                .rename_axis("Sentiment")
                .reset_index(name="Count")
            )
            st.bar_chart(dist_df.set_index("Sentiment"))

            st.subheader(f"Sampel Hasil Prediksi – {model_key}")
            st.dataframe(df_model.head(20), use_container_width=True)

    c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 1])

    with c2:
        st.download_button(
            "Download Hasil Prediksi",
            data=st.session_state.csv_pred,
            file_name=(
                f"{st.session_state.app_id}_prediksi_ulasan.xlsx"
                if st.session_state.is_combo else
                f"{st.session_state.app_id}_prediksi_ulasan.csv"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if st.session_state.is_combo else
                "text/csv"
            ),
            type="primary",
            key="dl_pred",
        )

    with c4:
        st.download_button(
            "Download Distribusi Sentimen",
            data=st.session_state.csv_dist,
            file_name=(
                f"{st.session_state.app_id}_distribusi_sentimen.xlsx"
                if st.session_state.is_combo else
                f"{st.session_state.app_id}_distribusi_sentimen.csv"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if st.session_state.is_combo else
                "text/csv"
            ),
            type="primary",
            key="dl_dist",
        )

    # ===== METRIK (dibanding rating bintang) =====
    def rating_to_label(r):
        if pd.isna(r):
            return None
        return 1 if r >= 4 else 0

    all_metrics = []
    for model_key, df_model in st.session_state.results.items():
        df_eval = df_model.copy()
        df_eval["true_label"] = df_eval["rating"].apply(rating_to_label)
        df_eval = df_eval.dropna(subset=["true_label"])

        if df_eval.empty:
            continue

        acc = accuracy_score(df_eval["true_label"], df_eval["pred"])
        prec = precision_score(df_eval["true_label"], df_eval["pred"])
        rec = recall_score(df_eval["true_label"], df_eval["pred"])
        f1  = f1_score(df_eval["true_label"], df_eval["pred"])

        all_metrics.append({
            "Model": model_key,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
        })

    if all_metrics:
        st.subheader("Perbandingan Metrik Evaluasi (dibanding rating bintang)")
        st.dataframe(pd.DataFrame(all_metrics), use_container_width=True)
    else:
        st.info("Tidak ada metrik yang bisa dihitung.")

elif page == "prediksi" and not st.session_state.results:
    st.info("Masukkan link/package, lalu klik **Prediksi**.")
