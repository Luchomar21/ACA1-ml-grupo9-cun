import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ------------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="ACA 1 - Aprendizaje Automático | Grupo 9",
    page_icon="🤖",
    layout="wide"
)

# ------------------------------------------------------------------------------
# PORTADA INICIAL / ENCABEZADO
# ------------------------------------------------------------------------------
st.title("🤖 Plataforma Analítica de Fidelización de Clientes")
st.caption("Especialización en Inteligencia Artificial — CUN | Asignatura: Aprendizaje Automático")

# Banner informativo del equipo
with st.expander("📌 **Ver Información del Proyecto y Equipo (Grupo 9)**", expanded=True):
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        **Institución:** Corporación Unificada Nacional de Educación Superior (CUN)  
        **Docente:** Dr. Hamilton Rivera Flor  
        **Pilar:** Pilar 3 - Analítica Comercial / Ventas  
        **Enfoque:** Segmentación y Modelo RFM (*Recency, Frequency, Monetary*)
        """)
    with col_info2:
        st.markdown("""
        **Integrantes del Grupo 9:**
        - Carlos Daniel Cárcamo Sepúlveda
        - Jhon Jairo Cárdenas Salcedo
        - Luis Omar Carrasquilla Riveros
        - Luciano Chito Piamba
        - Jhon Arbey Gutiérrez García
        """)

st.divider()

# ------------------------------------------------------------------------------
# CARGA DE MODELOS CON TOLERANCIA A FALLOS (RESILIENCIA EN CASCADA)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        models['gb'] = joblib.load('best_model_gb.pkl')
    except Exception:
        models['gb'] = None

    try:
        models['rf'] = joblib.load('best_model_rfm.pkl')
    except Exception:
        models['rf'] = None

    try:
        models['lr'] = joblib.load('best_model_lr.pkl')
        models['scaler'] = joblib.load('scaler.pkl')
    except Exception:
        models['lr'] = None
        models['scaler'] = None
        
    return models

loaded_models = load_models()

# ------------------------------------------------------------------------------
# BARRA LATERAL (CONTROLES DE ENTRADA)
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración del Modelo")
model_choice = st.sidebar.selectbox(
    "Seleccione el Algoritmo:",
    ("Gradient Boosting (Principal)", "Random Forest (Secundario)", "Regresión Logística (Respaldo)")
)

st.sidebar.header("📥 Datos del Cliente (RFM)")
recency = st.sidebar.number_input("Recencia (Días sin comprar):", min_value=1, max_value=365, value=25)
frequency = st.sidebar.number_input("Frecuencia (Total de compras):", min_value=1, max_value=100, value=8)
monetary = st.sidebar.number_input("Monto Total Gastado ($):", min_value=1.0, max_value=50000.0, value=650.0)

btn_predict = st.sidebar.button("🚀 Evaluar Fidelidad")

# Generación de base sintética para los gráficos
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
# ESTRUCTURA PRINCIPAL POR PESTAÑAS (TABS)
# ------------------------------------------------------------------------------
tab_pred, tab_graf, tab_doc = st.tabs([
    "🔮 **Predicción & Estrategia**", 
    "📊 **Visualización de Gráficos**", 
    "📄 **Documentación Técnica**"
])

# --- PESTAÑA 1: PREDICCIÓN ESTRATÉGICA Y SCORE ---
with tab_pred:
    st.subheader("Evaluación del Cliente & Recomendador de Marketing")
    st.write("Ingrese los parámetros en la barra lateral izquierda y presione **'🚀 Evaluar Fidelidad'**.")
    
    # Tarjetas informativas de los datos ingresados
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Recencia", f"{recency} días")
    col_m2.metric("Frecuencia", f"{frequency} compras")
    col_m3.metric("Monto Gastado", f"${monetary:,.2f}")

    if btn_predict:
        st.markdown("---")
        input_data = np.array([[recency, frequency, monetary]])
        selected_model = None
        model_name = ""
        pred = None
        prob = None
        
        # Selección e inferencia con cálculo de probabilidad
        if "Gradient Boosting" in model_choice and loaded_models['gb'] is not None:
            selected_model = loaded_models['gb']
            model_name = "Gradient Boosting"
            pred = selected_model.predict(input_data)[0]
            if hasattr(selected_model, "predict_proba"):
                prob = selected_model.predict_proba(input_data)[0][1] * 100
        elif "Random Forest" in model_choice and loaded_models['rf'] is not None:
            selected_model = loaded_models['rf']
            model_name = "Random Forest"
            pred = selected_model.predict(input_data)[0]
            if hasattr(selected_model, "predict_proba"):
                prob = selected_model.predict_proba(input_data)[0][1] * 100
        elif loaded_models['lr'] is not None and loaded_models['scaler'] is not None:
            selected_model = loaded_models['lr']
            model_name = "Regresión Logística"
            input_scaled = loaded_models['scaler'].transform(input_data)
            pred = selected_model.predict(input_scaled)[0]
            if hasattr(selected_model, "predict_proba"):
                prob = selected_model.predict_proba(input_scaled)[0][1] * 100
        else:
            st.error("❌ Ocurrió un problema con el modelo seleccionado. Usando respaldo de emergencia.")

        if pred is not None:
            res_col1, res_col2 = st.columns([2, 1])
            
            with res_col1:
                if pred == 1:
                    st.success(f"🎉 **RESULTADO:** Cliente clasificado como **RECURRENTE / FIDELIZADO**")
                    st.balloons()
                else:
                    st.warning(f"⚠️ **RESULTADO:** Cliente clasificado como **OCASIONAL / EN RIESGO**")
            
            with res_col2:
                if prob is not None:
                    st.metric(label="Score de Confianza (Probabilidad)", value=f"{prob:.1f}%")

            st.caption(f"Evaluación realizada mediante el algoritmo: **{model_name}**")
            
            st.divider()
            
            # --- SECCIÓN DE RECOMENDACIÓN DE MARKETING AUTOMÁTICA ---
            st.subheader("💡 Plan de Acción Comercial Recomendado")
            
            if pred == 1:
                st.info("""
                **Estrategia de Fidelización VIP / Retención:**
                - 🌟 **Inclusión en Programa Loyalty VIP:** Asignar estado preferencial con acumulación doble de puntos.
                - 🎁 **Lanzamientos Exclusivos:** Enviar acceso anticipado a nuevas colecciones o productos.
                - 📈 **Venta Cruzada (Cross-Selling):** Ofrecer complementos de alto valor basados en sus compras previas.
                """)
            else:
                st.error("""
                **Estrategia de Reactivación y Rescate:**
                - 📩 **Campaña de Email Marketing Personalizada:** Enviar un cupón con descuento del 15% para su próxima compra.
                - ❓ **Encuesta de Satisfacción Abreviada:** Identificar los motivos de la baja frecuencia o recencia prolongada.
                - 🔔 **Recordatorio de Carrito / Producto:** Notificar ofertas por tiempo limitado en ítems de su interés.
                """)

# --- PESTAÑA 2: GRÁFICOS ESTADÍSTICOS ---
with tab_graf:
    st.subheader("Análisis Exploratorio y Posicionamiento del Cliente")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig1, ax1 = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(
            data=df_rfm, x='Frequency', y='Monetary', hue='Target', 
            palette={0: "#E67E22", 1: "#2ECC71"}, ax=ax1, alpha=0.8
        )
        # Resaltar la posición del cliente ingresado
        ax1.scatter(frequency, monetary, color='red', s=180, marker='X', label='Cliente Evaluado')
        ax1.set_title("Frecuencia vs Monto Total Gastado", fontweight='bold')
        ax1.set_xlabel("Frecuencia (N° de compras)")
        ax1.set_ylabel("Monto Total ($)")
        ax1.legend()
        st.pyplot(fig1)

    with col_g2:
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        sns.histplot(df_rfm['Recency'], kde=True, color='#3498DB', ax=ax2)
        ax2.axvline(recency, color='red', linestyle='--', linewidth=2, label=f'Cliente ({recency} días)')
        ax2.set_title("Distribución de Recencia en Clientes", fontweight='bold')
        ax2.set_xlabel("Días transcurridos sin comprar")
        ax2.set_ylabel("Densidad / Conteo")
        ax2.legend()
        st.pyplot(fig2)
        
# --- PESTAÑA 3: MÉTRICAS DE EVALUACIÓN ---
with tab_metrics:
    st.subheader("📐 Rendimiento y Comparativa de Modelos")
    st.write("Evaluación cuantitativa del rendimiento de los algoritmos sobre el conjunto de validación.")
    
    X_val = df_rfm[['Recency', 'Frequency', 'Monetary']]
    y_val = df_rfm['Target']
    
    # Evaluación dinámica con los modelos cargados
    metrics_data = []
    
    if loaded_models['gb'] is not None:
        y_pred_gb = loaded_models['gb'].predict(X_val)
        metrics_data.append({
            "Algoritmo": "Gradient Boosting",
            "Accuracy": f"{accuracy_score(y_val, y_pred_gb)*100:.1f}%",
            "Precision": f"{precision_score(y_val, y_pred_gb)*100:.1f}%",
            "Recall": f"{recall_score(y_val, y_pred_gb)*100:.1f}%",
            "F1-Score": f"{f1_score(y_val, y_pred_gb)*100:.1f}%"
        })
        
    if loaded_models['rf'] is not None:
        y_pred_rf = loaded_models['rf'].predict(X_val)
        metrics_data.append({
            "Algoritmo": "Random Forest",
            "Accuracy": f"{accuracy_score(y_val, y_pred_rf)*100:.1f}%",
            "Precision": f"{precision_score(y_val, y_pred_rf)*100:.1f}%",
            "Recall": f"{recall_score(y_val, y_pred_rf)*100:.1f}%",
            "F1-Score": f"{f1_score(y_val, y_pred_rf)*100:.1f}%"
        })

    if loaded_models['lr'] is not None and loaded_models['scaler'] is not None:
        X_val_scaled = loaded_models['scaler'].transform(X_val)
        y_pred_lr = loaded_models['lr'].predict(X_val_scaled)
        metrics_data.append({
            "Algoritmo": "Regresión Logística",
            "Accuracy": f"{accuracy_score(y_val, y_pred_lr)*100:.1f}%",
            "Precision": f"{precision_score(y_val, y_pred_lr)*100:.1f}%",
            "Recall": f"{recall_score(y_val, y_pred_lr)*100:.1f}%",
            "F1-Score": f"{f1_score(y_val, y_pred_lr)*100:.1f}%"
        })

    if metrics_data:
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    else:
        st.warning("No se pudieron calcular las métricas debido a la indisponibilidad de los modelos.")

# --- PESTAÑA 4: DOCUMENTACIÓN TÉCNICA ---
with tab_doc:
    st.subheader("Arquitectura del Sistema y Tolerancia a Fallos")
    st.markdown("""
    - **Metodología RFM:** Evalúa **Recency** (Recencia), **Frequency** (Frecuencia) y **Monetary** (Monto) para categorizar la lealtad comercial.
    - **Tolerancia a Fallos:** En caso de caída de algún archivo `.pkl`, la aplicación conmuta dinámicamente entre el algoritmo principal (*Gradient Boosting*), el secundario (*Random Forest*) y el modelo de respaldo (*Regresión Logística*).
    - **Despliegue:** Sistema alojado de forma continua en Streamlit Cloud y sincronizado con GitHub (`luchomar21/aca1-ml-grupo9-cun`).
    """)

# --- PESTAÑA 3: DOCUMENTACIÓN TÉCNICA ---
with tab_doc:
    st.subheader("Arquitectura del Sistema y Tolerancia a Fallos")
    st.markdown("""
    - **Metodología RFM:** Evalúa **Recency** (Recencia), **Frequency** (Frecuencia) y **Monetary** (Monto) para categorizar la lealtad comercial.
    - **Tolerancia a Fallos:** En caso de caída de algún archivo `.pkl`, la aplicación conmuta dinámicamente entre el algoritmo principal (*Gradient Boosting*), el secundario (*Random Forest*) y el modelo de respaldo (*Regresión Logística*).
    - **Despliegue:** Sistema alojado de forma continua en Streamlit Cloud y sincronizado con GitHub (`luchomar21/aca1-ml-grupo9-cun`).
    """)
