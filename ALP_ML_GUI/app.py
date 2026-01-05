import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Smart Fairness Audit",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (FIX DROPDOWN ICON) ---
st.markdown("""
<style>
    /* Import Font Poppins */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Background Utama */
    .stApp {
        background-color: #0E1117;
        font-family: 'Poppins', sans-serif;
    }

    /* PENTING: Hanya target elemen TEKS spesifik.
       JANGAN masukkan 'span' atau 'i' ke sini agar icon panah tidak rusak.
    */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stButton button, small {
        font-family: 'Poppins', sans-serif !important;
        color: #E0E0E0;
    }
    
    /* Judul lebih terang */
    h1, h2, h3, strong {
        color: #FFFFFF !important;
    }

    /* Input Fields & Dropdown Background */
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput div[data-baseweb="input"] > div {
        background-color: #FFFFFF;
        color: #000000;
        border-color: #4A4A4A;
    }
    
    /* Warna teks pilihan di dalam dropdown */
    .stSelectbox div[data-baseweb="select"] span {
        color: white; 
    }

    /* Kartu Metrik */
    div.metric-card {
        background-color: #262730;
        border: 1px solid #363945;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        transition: transform 0.2s;
    }
    div.metric-card:hover {
        transform: translateY(-5px);
        border-color: #4CAF50;
        box-shadow: 0 8px 15px rgba(0,0,0,0.5);
    }

    /* Tombol Custom */
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1B5E20;
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.4);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161920;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 5px;
        color: white;
        font-family: 'Poppins', sans-serif !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2E7D32 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('best_rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, scaler, model_columns
    except FileNotFoundError:
        return None, None, None

model, scaler, model_columns = load_assets()

# --- 4. HELPER FUNCTIONS ---
EDUCATION_MAP = {
    'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, 
    '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 
    'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14, 
    'Prof-school': 15, 'Doctorate': 16
}

def preprocess_input(data, model_columns, scaler):
    df_input = pd.DataFrame(columns=model_columns)
    df_input.loc[0] = 0
    
    sex_val = 1 if data['sex'] == 'Female' else 0
    race_val = 1 if data['race'] == 'White' else 0
    is_married_val = 1 if data['marital_status'].startswith('Married') else 0
    rel_val = 1 if data['relationship'] in ['Husband', 'Wife'] else 0
    
    numerical_features = [
        'age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week',
        'sex', 'is_married', 'relationship_status', 'race_grouped'
    ]
    
    raw_numerics = pd.DataFrame([[
        data['age'], EDUCATION_MAP[data['education']], data['capital_gain'], 
        data['capital_loss'], data['hours_per_week'], sex_val, 
        is_married_val, rel_val, race_val
    ]], columns=numerical_features)
    
    scaled_numerics = scaler.transform(raw_numerics)
    
    for i, col in enumerate(numerical_features):
        if col in df_input.columns:
            df_input.loc[0, col] = scaled_numerics[0, i]
            
    ohe_categories = {
        'workclass': data['workclass'], 'education': data['education'],
        'occupation': data['occupation'], 'native.country': data['native_country']
    }
    
    for prefix, value in ohe_categories.items():
        col_name = f"{prefix}_{value}"
        if col_name in df_input.columns:
            df_input.loc[0, col_name] = 1
            
    return df_input

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🛡️ Fairness Audit")
    st.write("Machine Learning System")
    
    menu = st.radio("", ["Dashboard", "Simulation", "Fairness Check"], index=0)
    
    st.markdown("---")
    st.markdown("""
    <div style='background-color: #262730; padding: 10px; border-radius: 8px; font-size: 12px; border: 1px solid #363945; color: #E0E0E0;'>
        <strong>Model Info:</strong><br>
        • Random Forest (Tuned)<br>
        • Dataset: Adult 1994<br>
        • Status: <span style='color:#4CAF50'>Ready</span>
    </div>
    """, unsafe_allow_html=True)

# --- 6. PAGE: DASHBOARD ---
if menu == "Dashboard":
    st.markdown("<h1>Income Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("Memantau performa dan keadilan model AI dalam prediksi pendapatan individu.")
    
    st.info("🎯 Tujuan Sistem: Memprediksi apakah pendapatan >$50K/tahun sekaligus mengaudit bias gender & ras.")

    # Cards
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("Akurasi Model", "81%", "✅ Stabil", "#4CAF50"),
        ("F1-Score", "0.77", "⚖️ Seimbang", "#2196F3"),
        ("Recall (>50K)", "85%", "🚀 Tinggi", "#FFC107"),
        ("Bias Gender", "Terdeteksi", "⚠️ Waspada", "#FF5252")
    ]
    
    for col, (label, val, status, color) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <h2 style="margin:0; color:{color}; font-family: 'Poppins', sans-serif;">{val}</h2>
                <p style="margin:5px 0 0 0; font-size:14px; color:#B0B0B0;">{label}</p>
                <small style="font-weight:bold; color:#FAFAFA;">{status}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📊 Ringkasan Performa")
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        imp_data = pd.DataFrame({
            'Fitur': ['Relationship', 'Capital Gain', 'Age', 'Education', 'Hours/Week', 'Capital Loss'],
            'Pentingnya': [0.050, 0.037, 0.021, 0.020, 0.008, 0.005]
        }).sort_values('Pentingnya', ascending=True)
        
        fig = px.bar(imp_data, x='Pentingnya', y='Fitur', orientation='h', 
                     title="Faktor Penentu Keputusan AI", 
                     color='Pentingnya',
                     color_continuous_scale='Greens',
                     template='plotly_dark')
        
        fig.update_layout(
            font=dict(family="Poppins, sans-serif", size=12, color="white"),
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_chart2:
        st.write("#### Matriks Prediksi")
        cm_data = pd.DataFrame({"<=50K": [3935, 243], ">50K": [1005, 1325]}, 
                               index=["Aktual Low", "Aktual High"])
        st.dataframe(cm_data, use_container_width=True)
        st.caption("Model dioptimalkan untuk menangkap kelompok High Income (Recall Tinggi).")

# --- 7. PAGE: SIMULATION ---
elif menu == "Simulation":
    st.header("🔮 Simulator Profil")
    st.write("Masukkan data profil di bawah ini untuk melihat prediksi AI.")
    
    if model is None:
        st.error("⚠️ Model belum ditemukan! Jalankan `generate_model.py` dulu.")
    else:
        with st.container():
            c1, c2, c3 = st.columns(3)
            with c1:
                age = st.number_input("Usia", 17, 90, 30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                race = st.selectbox("Ras", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
            with c2:
                education = st.selectbox("Pendidikan", list(EDUCATION_MAP.keys()), index=9)
                workclass = st.selectbox("Sektor Kerja", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov"])
                hours = st.slider("Jam Kerja/Minggu", 1, 100, 40)
            with c3:
                occupation = st.selectbox("Pekerjaan", ["Exec-managerial", "Prof-specialty", "Sales", "Craft-repair", "Tech-support", "Adm-clerical", "Other-service"])
                marital = st.selectbox("Status Nikah", ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed"])
                relationship = st.selectbox("Peran Keluarga", ["Husband", "Wife", "Own-child", "Unmarried", "Not-in-family"])
                
        with st.expander("Aset Finansial (Opsional)"):
            ec1, ec2, ec3 = st.columns(3)
            cap_gain = ec1.number_input("Capital Gain", 0, 100000, 0)
            cap_loss = ec2.number_input("Capital Loss", 0, 100000, 0)
            native_country = ec3.selectbox("Negara Asal", ["United-States", "Mexico", "Philippines", "Germany", "Other"])

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Analisis Profil"):
            with st.spinner("Menganalisis pola data..."):
                input_data = {
                    'age': age, 'sex': gender, 'race': race, 'native_country': native_country,
                    'workclass': workclass, 'education': education, 'occupation': occupation,
                    'hours_per_week': hours, 'marital_status': marital, 
                    'relationship': relationship, 'capital_gain': cap_gain, 'capital_loss': cap_loss
                }
                
                try:
                    final_input = preprocess_input(input_data, model_columns, scaler)
                    pred = model.predict(final_input)[0]
                    prob = model.predict_proba(final_input)[0][1]
                    
                    st.divider()
                    col_res, col_gauge = st.columns([1, 2])
                    
                    with col_res:
                        st.markdown("### Hasil Prediksi")
                        if pred == 1:
                            st.markdown("""
                            <div style="background-color: #1B5E20; color: white; padding: 20px; border-radius: 10px; border: 1px solid #4CAF50; text-align: center;">
                                <h1 style="color: white !important; margin:0; font-family: 'Poppins', sans-serif;">>$50K</h1>
                                <p style="margin:0;">High Income</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style="background-color: #7F0000; color: white; padding: 20px; border-radius: 10px; border: 1px solid #FF5252; text-align: center;">
                                <h1 style="color: white !important; margin:0; font-family: 'Poppins', sans-serif;"><=$50K</h1>
                                <p style="margin:0;">Low/Mid Income</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with col_gauge:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob * 100,
                            title = {'text': "Probabilitas High Income (%)", 'font': {'color': 'white', 'family': 'Poppins'}},
                            number = {'font': {'color': 'white', 'family': 'Poppins'}},
                            gauge = {
                                'axis': {'range': [0, 100], 'tickcolor': "white"},
                                'bar': {'color': "#4CAF50" if pred == 1 else "#FF5252"},
                                'bgcolor': "#262730",
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 50], 'color': "#333"},
                                    {'range': [50, 100], 'color': "#444"}
                                ]
                            }
                        ))
                        fig.update_layout(
                            height=250, 
                            margin=dict(t=30,b=10,l=20,r=20),
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color="white", family="Poppins")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error: {e}")

# --- 8. PAGE: FAIRNESS CHECK ---
elif menu == "Fairness Check":
    st.header("⚖️ Audit Keadilan Model")
    
    tab1, tab2 = st.tabs(["Gender Bias", "Racial Bias"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            df_sex = pd.DataFrame({
                "Gender": ["Male", "Male", "Female", "Female"],
                "Metrik": ["Recall", "Precision", "Recall", "Precision"],
                "Skor": [0.86, 0.56, 0.76, 0.62]
            })
            fig = px.bar(df_sex, x="Gender", y="Skor", color="Metrik", barmode="group",
                         title="Performa Model: Pria vs Wanita", 
                         color_discrete_sequence=["#42A5F5", "#AB47BC"],
                         template='plotly_dark')
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins, sans-serif", color="white")
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            st.markdown("""
            <div style="background-color: #3E2723; padding: 15px; border-radius: 8px; border-left: 4px solid #FF5252; color: #E0E0E0;">
                <strong style="color: #FF5252;">⚠️ Temuan Bias:</strong><br>
                Model memiliki <strong>Recall</strong> lebih rendah pada Wanita (76%) dibandingkan Pria (86%).<br><br>
                Artinya: Wanita kaya lebih sering <em>gagal dideteksi</em> oleh model dibandingkan Pria kaya.
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        categories = ['Recall', 'Precision', 'F1-Score']
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[0.85, 0.57, 0.68], theta=categories, fill='toself', name='White',
            line_color='#66BB6A'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[0.79, 0.52, 0.62], theta=categories, fill='toself', name='Non-White',
            line_color='#FF7043'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], color='white'),
                bgcolor='#262730'
            ),
            title="Perbandingan Metrik: White vs Non-White",
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Poppins, sans-serif", color="white")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info("Kelompok White memiliki performa deteksi yang lebih unggul di semua metrik.")