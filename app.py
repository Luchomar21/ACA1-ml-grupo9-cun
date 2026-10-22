# ==============================================================================
# ESPECIALIZACIÓN EN INTELIGENCIA ARTIFICIAL - CUN
# ASIGNATURA: APRENDIZAJE AUTOMÁTICO | DOCENTE: DR. HAMILTON RIVERA FLOR
# PROYECTO ACA 1 - GRUPO 9: APLICACIÓN INTERACTIVA CON TOLERANCIA A FALLOS
# Integrantes: Carlos Cárcamo, Jhon Cárdenas, Luis Carrasquilla, Luciano Chito, Jhon Gutiérrez
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuración inicial de la página
st.set_page_config(
    page_title="Predicción de Fidelidad de Cliente - CUN Grupo 9",
    page_icon="🛒",
    layout="wide"
)

# ------------------------------------------------------------------------------
# 1. ENCABEZADO OFICIAL
# ------------------------------------------------------------------------------
st.title("🛒 Sistema Predictivo Multimodelo con Tolerancia a Fallos")
st.markdown("""
**Corporación Unificada Nacional de Educación Superior (CUN)**  
*Especialización en Inteligencia Artificial* — **Asignatura:** Aprendizaje Automático  
**Docente:** Dr. Hamilton Rivera Flor | **Grupo 9**  
**Integrantes:** Carlos Cárcamo, Jhon Cárdenas, Luis Carrasquilla, Luciano Chito, Jhon Gutiérrez.
""")
st.divider()

# ------------------------------------------------------------------------------
# 2. CARGA ROBUSTA DE MÚLTIPLES MODELOS (SISTEMA DE FALLBACK / RESPALDO)
# ------------------------------------------------------------------------------
# Archivos de modelos que la app intentará buscar en orden de preferencia
MODEL_FILES = {
    "Gradient Boosting": "best_model_gb.pkl",
    "Random Forest": "best_model_rfm.pkl",
    "Regresión Logística": "best_model_lr.pkl"
}

@st.cache_resource
def load_all_models_and_scaler():
    loaded_models = {}
    scaler = None
    
    # 1. Cargar Escalador Z-score
    try:
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
    except Exception as e:
        st.warning(f"No se pudo cargar el escalador: {e}")

    # 2. Intento de carga de modelos uno por uno
    for model_name, filename in MODEL_FILES.items():
        if os.path.exists(filename):
            try:
                loaded_models[model_name] = joblib.load(filename)
            except Exception as e:
                st.warning(f"Error al cargar el archivo '{filename}': {e}")
    
    return loaded_models, scaler

loaded_models, scaler = load_all_models_and_scaler()

# ------------------------------------------------------------------------------
# 3. CONTROLES EN EL PANEL LATERAL (SIDEBAR)
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración del Modelo")

if loaded_models:
    # Selector manual de modelo disponible
    selected_model_name = st.sidebar.selectbox(
        "Seleccione el algoritmo a utilizar:",
        options=list(loaded_models.keys()),
        help="Permite intercambiar dinámicamente entre los modelos entrenados."
    )
    active_model = loaded_models[selected_model_name]
    st.sidebar.success(f"✔ Modelo activo: **{selected_model_name}**")
else:
    active_model = None
    st.sidebar.error("⚠️ No se encontraron archivos `.pkl` de modelos. La app operará en Modo Simulación.")

st.sidebar.divider()
st.sidebar.header("📊 Parámetros del Cliente (Modelo RFM)")

recency = st.sidebar.slider(
    "1. Recencia (Días desde la última compra):",
    min_value=1, max_value=365, value=30,
    help="Días transcurridos desde la última transacción del cliente."
)

frequency = st.sidebar.number_input(
    "2. Frecuencia (Número total de compras):",
    min_value=1, max_value=100, value=3, step=1,
    help="Cantidad total de órdenes o facturas generadas."
)

monetary = st.sidebar.number_input(
    "3. Monto Total Gastado (USD):",
    min_value=1.0, max_value=50000.0, value=450.0, step=10.0,
    help="Suma económica acumulada consumida por el cliente."
)

st.sidebar.divider()
predict_btn = st.sidebar.button("🔮 Generar Predicción", type="primary")

# ------------------------------------------------------------------------------
# 4. FUNCIÓN PREDICTIVA CON TOLERANCIA A FALLOS (FALLBACK AUTOMÁTICO)
# ------------------------------------------------------------------------------
def execute_prediction_with_fallback(input_df):
    """
    Intenta predecir con el modelo seleccionado. Si falla por algún motivo técnico,
    pasa automáticamente al siguiente modelo disponible en cascada.
    """
    # Orden de prueba: primero el seleccionado, luego el resto
    trial_order = [selected_model_name] + [m for m in loaded_models.keys() if m != selected_model_name]
    
    for model_name in trial_order:
        try:
            model = loaded_models[model_name]
            
            # Aplicar escalamiento si el modelo es Regresión Logística
            if model_name == "Regresión Logística" and scaler is not None:
                input_data = scaler.transform(input_df)
            else:
                input_data = input_df
                
            prediction = model.predict(input_data)[0]
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0][1]
            else:
                proba = 0.5
                
            return prediction, proba, model_name, None
        except Exception as e:
            # Si falla, continúa el bucle e intenta con el siguiente modelo
            continue
            
    # Regla Heurística de Respaldo Final (Si fallan todos los archivos .pkl)
    # Ejemplo: Si compra más de 2 veces y hace menos de 90 días, es recurrente
    fallback_pred = 1 if (input_df['Frequency'].values[0] > 2 and input_df['Recency'].values[0] <= 90) else 0
    fallback_proba = 0.85 if fallback_pred == 1 else 0.20
    return fallback_pred, fallback_proba, "Regla Lógica de Respaldo (Fallback)", "Todos los modelos serializados fallaron o no fueron encontrados."

# ------------------------------------------------------------------------------
# 5. PESTAÑAS Y DESPLIEGUE DE RESULTADOS
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Predicción en Tiempo Real", "🛡️ Arquitectura de Fallback", "ℹ️ Información del Proyecto"])

with tab1:
    st.subheader("Diagnóstico y Clasificación del Cliente")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Recencia", f"{recency} Días")
    c2.metric("Frecuencia", f"{frequency} Compras")
    c3.metric("Monto Total", f"${monetary:,.2f}")
    
    st.write("---")
    
    if predict_btn:
        input_df = pd.DataFrame({'Recency': [recency], 'Frequency': [frequency], 'Monetary': [monetary]})
        
        prediction, proba, used_model_name, warning_msg = execute_prediction_with_fallback(input_df)
        
        if warning_msg:
            st.warning(f"⚠️ **Aviso de Tolerancia a Fallos:** {warning_msg}")
            
        st.caption(f"🤖 **Procesado mediante:** `{used_model_name}`")
        
        if prediction == 1:
            st.success("### ✅ Categoría: CLIENTE RECURRENTE / FIDELIZADO")
            st.write(f"**Probabilidad Estimada de Recurrencia:** `{proba * 100:.2f}%`")
            st.info("💡 **Estrategia Recomendada:** Programas de lealtad, venta cruzada (*cross-selling*) y beneficios exclusivos.")
        else:
            st.warning("### ⚠️ Categoría: CLIENTE OCASIONAL / EN RIESGO DE DESERCIÓN")
            st.write(f"**Probabilidad Estimada de Recurrencia:** `{proba * 100:.2f}%`")
            st.error("💡 **Estrategia Recomendada:** Cupones de reactivación, campañas de retención y encuestas de servicio.")
    else:
        st.info("👈 Haga clic en **'Generar Predicción'** para calcular la clasificación.")

with tab2:
    st.subheader("Mecanismo de Tolerancia a Fallos (Cascada / Fallback)")
    st.markdown("""
    Esta aplicación implementa un patrón de **Resiliencia y Tolerancia a Fallos**:
    
    1. **Nivel 1 (Modelo Preferente):** Ejecuta la predicción con el modelo seleccionado por el usuario en el panel lateral (*Gradient Boosting / Random Forest / Regresión Logística*).
    2. **Nivel 2 (Cascada Automática):** Si el modelo seleccionado falla por inconsistencia en memoria o corrupción de datos, la app salta automáticamente al segundo mejor modelo disponible sin interrumpir la experiencia del usuario.
    3. **Nivel 3 (Respaldo Heurístico):** En caso extremo de ausencia total de archivos `.pkl`, la aplicación activa un motor de reglas del negocio basado en parámetros RFM para garantizar la continuidad del servicio.
    """)

with tab3:
    st.subheader("Proyecto ACA 1 - Especialización en IA (CUN)")
    st.markdown("""
    * **Docente:** Dr. Hamilton Rivera Flor[cite: 1]
    * **Equipo 9:** Carlos Cárcamo, Jhon Cárdenas, Luis Carrasquilla, Luciano Chito, Jhon Gutiérrez.
    * **Pilar:** Ventas y Analítica Comercial[cite: 2]
    * **Técnica:** Modelado Supervisado Clásico sobre Métricas RFM (Recency, Frequency, Monetary).[cite: 2, 5, 6]
    """)
