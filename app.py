import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuración de página
st.set_page_config(
    page_title="Predicción de Fidelidad - Grupo 9 CUN",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Clasificación de Clientes (Modelo RFM)")
st.markdown("### ACA 1 - Aprendizaje Automático | Grupo 9 - CUN")

# ------------------------------------------------------------------------------
# 1. CARGA DE MODELOS CON TOLERANCIA A FALLOS (RESILIENCIA EN CASCADA)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    
    # Intento 1: Gradient Boosting
    try:
        models['gb'] = joblib.load('best_model_gb.pkl')
    except Exception as e:
        models['gb'] = None

    # Intento 2: Random Forest
    try:
        models['rf'] = joblib.load('best_model_rfm.pkl')
    except Exception as e:
        models['rf'] = None

    # Intento 3: Regresión Logística y Scaler
    try:
        models['lr'] = joblib.load('best_model_lr.pkl')
        models['scaler'] = joblib.load('scaler.pkl')
    except Exception as e:
        models['lr'] = None
        models['scaler'] = None
        
    return models

loaded_models = load_models()

# Selector de modelo en barra lateral
st.sidebar.header("⚙️ Configuración del Modelo")
model_choice = st.sidebar.selectbox(
    "Seleccione el Algoritmo a Evaluar:",
    ("Gradient Boosting (Principal)", "Random Forest (Secundario)", "Regresión Logística (Respaldo)")
)

# ------------------------------------------------------------------------------
# 2. ENTRADA DE DATOS DEL CLIENTE
# ------------------------------------------------------------------------------
st.sidebar.header("📥 Métricas del Cliente (RFM)")
recency = st.sidebar.number_input("Recencia (Días sin comprar):", min_value=1, max_value=365, value=30)
frequency = st.sidebar.number_input("Frecuencia (Número de compras):", min_value=1, max_value=100, value=5)
monetary = st.sidebar.number_input("Monto Total Gastado ($):", min_value=1.0, max_value=50000.0, value=450.0)

# Generación de datos sintéticos base para graficar y contextualizar
@st.cache_data
def get_sample_rfm():
    np.random.seed(42)
    rec = np.random.randint(1, 100, 200)
    freq = np.random.randint(1, 15, 200)
    mon = np.random.uniform(20, 2000, 200)
    target = (freq > 5).astype(int)
    return pd.DataFrame({'Recency': rec, 'Frequency': freq, 'Monetary': mon, 'Target': target})

df_rfm = get_sample_rfm()

# ------------------------------------------------------------------------------
# 3. PREDICCIÓN CON EL MODELO SELECCIONADO
# ------------------------------------------------------------------------------
st.subheader("🔮 Resultado de la Predicción")

if st.sidebar.button("Predecir Fidelidad"):
    input_data = np.array([[recency, frequency, monetary]])
    
    selected_model = None
    model_name = ""
    
    if "Gradient Boosting" in model_choice and loaded_models['gb'] is not None:
        selected_model = loaded_models['gb']
        model_name = "Gradient Boosting"
        pred = selected_model.predict(input_data)[0]
    elif "Random Forest" in model_choice and loaded_models['rf'] is not None:
        selected_model = loaded_models['rf']
        model_name = "Random Forest"
        pred = selected_model.predict(input_data)[0]
    elif loaded_models['lr'] is not None and loaded_models['scaler'] is not None:
        selected_model = loaded_models['lr']
        model_name = "Regresión Logística"
        input_scaled = loaded_models['scaler'].transform(input_data)
        pred = selected_model.predict(input_scaled)[0]
    else:
        st.error("❌ No se pudo cargar ningún modelo ejecutable.")
        pred = None

    if pred is not None:
        if pred == 1:
            st.success(f"✅ **CLIENTE RECURRENTE / FIDELIZADO** (Evaluado con {model_name})")
            st.balloons()
        else:
            st.warning(f"⚠️ **CLIENTE OCASIONAL / EN RIESGO** (Evaluado con {model_name})")

st.divider()

# ------------------------------------------------------------------------------
# 4. VISUALIZACIÓN DE GRÁFICOS ESTADÍSTICOS
# ------------------------------------------------------------------------------
st.subheader("📈 Análisis Exploratorio de Clientes (RFM)")

tab1, tab2 = st.tabs(["📊 Distribución de Frecuencia vs Monto", "📉 Distribución de Recencia"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=df_rfm, x='Frequency', y='Monetary', hue='Target', 
        palette={0: "orange", 1: "green"}, ax=ax1, alpha=0.8
    )
    # Marcar la posición del cliente consultado
    ax1.scatter(frequency, monetary, color='red', s=150, marker='X', label='Cliente Consultado')
    ax1.set_title("Relación Frecuencia vs Monto Total Gastado")
    ax1.set_xlabel("Frecuencia (Compras)")
    ax1.set_ylabel("Monto ($)")
    ax1.legend()
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(df_rfm['Recency'], kde=True, color='skyblue', ax=ax2)
    ax2.axvline(recency, color='red', linestyle='--', linewidth=2, label=f'Recencia Cliente ({recency} días)')
    ax2.set_title("Distribución de Recencia en la Base de Clientes")
    ax2.set_xlabel("Días transcurridos")
    ax2.legend()
    st.pyplot(fig2)
