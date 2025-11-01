import io
import re
import pandas as pd
import streamlit as st
import joblib
import base64
from pathlib import Path
from google_play_scraper import app as gp_app, reviews, Sort
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# PAGE CONFIG
st.set_page_config(
    page_title="UlasAnalisa – Prediksi Sentimen",
    page_icon="static/logo_ulas.png", 
    layout="wide",
    initial_sidebar_state="expanded",
)

for k, v in {
    "results": {},          # dict: {"SVM (RBF)": df, "RandomForest": df}
    "app_id": None,
    "csv_pred": None,       # bytes (csv/xlsx)
    "csv_dist": None,       # bytes (csv/xlsx)
    "is_combo": False,      # True saat pilih "Gabungan (SVM + RF)"
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
<style>
/* Paku tombol collapse/expand sidebar di pojok kiri-atas dan taruh di atas header kustom */
[data-testid="stSidebarCollapseButton"]{
  position: fixed !important;
  top: 12px !important;
  left: 12px !important;
  z-index: 10000 !important;
}

/* Kalau kamu punya header/navbar kustom yang fixed, jangan nutup area kiri-atas */
@media (max-width: 768px){
  /* Lebarin klik area tombol di mobile biar gampang ditekan */
  [data-testid="stSidebarCollapseButton"] button {
    padding: 8px 10px !important;
  }
  /* Biar sidebar nggak melebar full menutupi tombol */
  [data-testid="stSidebar"]{
    width: 78vw !important;
    min-width: 260px !important;
  }
}
</style>
""", unsafe_allow_html=True)

#LOGO base64
def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_left_b64 = img_to_base64("static/logo_ulas.png")
logo_right_b64 = img_to_base64("static/fti_untar.png")

#BACA PAGE DARI URL
try:  
    page = st.query_params.get("page", "home")
except AttributeError:  
    page = st.experimental_get_query_params().get("page", ["home"])[0]

home_active = "active" if page == "home" else ""
pred_active = "active" if page == "prediksi" else ""
tentang_active = "active" if page == "tentang" else ""

# NAVBAR
st.markdown(
    f"""
<style>
[data-testid="stHeader"] {{
    display: none;
}}
.navbar {{
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 80px;
    background: #ffffff;
    display: flex;
    align-items: center;
    padding: 0 1.5rem;
    border-bottom: 3px solid #b71c1c;
    z-index: 100000;
}}
/* konten turun 90px karena navbar fixed */
[data-testid="stAppViewContainer"] > .main {{
    margin-top: 90px;
}}

.nav-left, .nav-right {{
    width: 220px;
    display: flex;
    justify-content: center;
    align-items: center;
}}
.nav-center {{
    flex: 1;
    display: flex;
    justify-content: center;
    gap: 2.5rem;
}}
.nav-center a {{
    text-decoration: none;
    color: #444;
    font-weight: 500;
}}
.nav-center a.active {{
    color: #b71c1c;
    border-bottom: 2px solid #b71c1c;
    padding-bottom: 4px;
}}
.logo-left {{ height: 150px; }}
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
""",
    unsafe_allow_html=True,
)

if page == "prediksi":
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        position: fixed;
        top: 90px;
        left: 0;
        height: 100%;
        width: 18rem;
        z-index: 99999;
    }
    [data-testid="stAppViewContainer"] > .main {
        margin-left: 18rem;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stAppViewContainer"] > .main { margin-left: 0; }
    </style>
    """, unsafe_allow_html=True)


# HALAMAN BERANDA
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
    st.write(
        "Buka Google Play Store di HP → cari aplikasinya → ketuk **⋮ → Share** → pilih **Copy URL**."
    )

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

# HALAMAN TENTANG
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

#HALAMAN PREDIKSI
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
        st.error(
            f"Gagal memuat artifacts.\n"
            f"Detail: {e}"
        )
        st.stop()

    # Session state
    for k, v in dict(pred_df=None, dist_df=None, app_id=None, csv_pred=None, csv_dist=None).items():
        if k not in st.session_state:
            st.session_state[k] = v
            if "results" not in st.session_state:
    # results akan jadi dict: {"SVM (RBF)": df_svm, "RandomForest": df_rf}
                st.session_state.results = {}
            if "app_id" not in st.session_state:
                st.session_state.app_id = None
            if "csv_pred" not in st.session_state:
                st.session_state.csv_pred = None
            if "csv_dist" not in st.session_state:
                st.session_state.csv_dist = None

    # model
    avail = []
    if svm_model is not None:
        avail.append("SVM (RBF)")
    if rf_model is not None:
        avail.append("RandomForest")
    if svm_model is not None and rf_model is not None:
        avail.append("SVM dan RandomForest")
    if not avail:
        st.error("Tidak ada model yang tersedia.")
        st.stop()

    with st.sidebar:
        st.header("Pengaturan")
        model_name = st.selectbox("Pilih model", avail, index=0)
        lang = st.selectbox("Bahasa ulasan", ["id", "en"], index=0)
        country = st.selectbox("Negara", ["id", "us"], index=0)
        n_reviews = st.slider("Jumlah ulasan di-scrape", 50, 1000, 200, 50)
        sort_opt = st.selectbox("Urutkan", ["NEWEST", "MOST_RELEVANT"], index=0)
        run = st.button("Prediksi")

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

        with st.spinner(f"Mengambil {n_reviews} ulasan..."):
            df = scrape_reviews(app_id, lang=lang, country=country, n=n_reviews, sort=sort_opt)
        if df.empty:
            st.warning("Tidak ada ulasan yang diambil.")
            st.stop()

        df = df.rename(columns={"content": "text", "score": "rating", "at": "date"})
        cols = ["text","rating","date","userName","replyContent"]
        df = df[[c for c in cols if c in df.columns]].copy()

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

            else:
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


        st.session_state.results = results
        st.session_state.app_id = app_id
        

        if model_name == "SVM dan RandomForest":
            df_svm = results["SVM (RBF)"]
            df_rf  = results["RandomForest"]

            # bikin workbook di memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_svm.to_excel(writer, sheet_name="SVM (RBF)", index=False)
                df_rf.to_excel(writer, sheet_name="RandomForest", index=False)
            output.seek(0)

            # simpan ke session_state
            st.session_state.csv_pred = output.getvalue()   # isinya xlsx sekarang

            # distribusi juga 2 sheet
            dist_svm = (
                df_svm["pred_label"].value_counts()
                .rename_axis("sentiment")
                .reset_index(name="count")
            )
            dist_rf = (
                df_rf["pred_label"].value_counts()
                .rename_axis("sentiment")
                .reset_index(name="count")
            )

            dist_out = io.BytesIO()
            with pd.ExcelWriter(dist_out, engine="xlsxwriter") as writer:
                dist_svm.to_excel(writer, sheet_name="SVM (RBF)", index=False)
                dist_rf.to_excel(writer, sheet_name="RandomForest", index=False)
            dist_out.seek(0)

            st.session_state.csv_dist = dist_out.getvalue()

        else:
            # mode 1 model saja → tetap CSV
            first_key = next(iter(results))
            first_df = results[first_key]

            st.session_state.csv_pred = first_df.to_csv(index=False).encode("utf-8")
            dist_df = (
                first_df["pred_label"]
                .value_counts()
                .rename_axis("sentiment")
                .reset_index(name="count")
            )
            
            st.session_state.csv_dist = dist_df.to_csv(index=False).encode("utf-8")
            
if st.session_state.results:
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

    is_combo = st.session_state.get("is_combo", False)

    with c2:
        st.download_button(
            "Download Hasil Prediksi",
            data=st.session_state.csv_pred,
            file_name=(
                f"{st.session_state.app_id}_prediksi_ulasan.xlsx"
                if is_combo else
                f"{st.session_state.app_id}_prediksi_ulasan.csv"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if is_combo else
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
                if is_combo else
                f"{st.session_state.app_id}_distribusi_sentimen.csv"
            ),
            mime=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if is_combo else
                "text/csv"
            ),
            type="primary",
            key="dl_dist",
        )

    # METRIK
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
        f1 = f1_score(df_eval["true_label"], df_eval["pred"])

        all_metrics.append({
            "Model": model_key,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })

    if all_metrics:
        st.subheader("Perbandingan Metrik Evaluasi (dibanding rating bintang)")
        st.dataframe(pd.DataFrame(all_metrics), use_container_width=True)
    else:
        st.info("Tidak ada metrik yang bisa dihitung.")

else:
    st.info("Masukkan link/package, lalu klik **Prediksi**.")

        