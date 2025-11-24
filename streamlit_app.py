import streamlit as st
import time
import pandas as pd
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Sistema Multi-Agente de Salud Mental", layout="wide")

# --- 1. SIMULACIÓN DEL MODELO DE DEEP LEARNING (Backend) ---
# En la realidad, aquí cargarías tu modelo .h5 o .pt
def predecir_depresion_dl(texto):
    """
    Simula la predicción de tu modelo entrenado con el dataset de Reddit.
    Retorna: Probabilidad (0.0 a 1.0) y Etiqueta
    """
    # Lógica simulada basada en palabras clave (SOLO PARA PROTOTIPO)
    # Tu modelo real reemplazará esto.
    palabras_clave = ['triste', 'solo', 'ayuda', 'suicidio', 'fin', 'cansado', 'dolor', 'nada']
    score = 0.1
    for palabra in palabras_clave:
        if palabra in texto.lower():
            score += 0.15

    noise = np.random.uniform(0, 0.1) # Pequeña variación
    final_score = min(score + noise, 0.99)

    label = "Depresión" if final_score > 0.5 else "Normal"
    return final_score, label

# --- 2. SIMULACIÓN DEL AGENTE EXPLICABLE (XAI con Gemini) ---
def agente_xai_explicacion(texto, probabilidad):
    """
    Aquí iría la llamada a la API de Gemini (google-generativeai).
    Prompt: "Actúa como psicólogo. El modelo predijo X probabilidad de depresión. Explica por qué basándote en el texto."
    """
    # Respuesta simulada de la API
    if probabilidad > 0.5:
        return f"""
        **Análisis del Agente (XAI):**
        El modelo ha detectado un riesgo alto ({probabilidad:.2%}) debido a patrones lingüísticos preocupantes.
        1. **Léxico Absolutista:** El usuario usa términos que denotan desesperanza.
        2. **Foco Interno:** El texto muestra aislamiento social.
        **Recomendación:** Activar protocolo de prevención de recaídas.
        """
    else:
        return f"""
        **Análisis del Agente (XAI):**
        El texto se clasifica dentro de parámetros normales ({probabilidad:.2%}).
        Aunque expresa emociones, no muestra patrones clínicos de recaída inminente según el entrenamiento del dataset Reddit.
        """

# --- 3. INTERFAZ GRÁFICA (Streamlit) ---

# Título y Descripción
st.title("🧠 Sistema Multi-Agente: Predicción de Recaídas")
st.markdown("""
Este sistema monitorea la actividad en redes sociales (simulación Reddit)
y utiliza **Deep Learning + Agentes Cognitivos** para detectar signos tempranos de depresión.
""")

# Dividir la pantalla en dos columnas
col_paciente, col_sistema = st.columns([1, 1])

# --- COLUMNA IZQUIERDA: SIMULACIÓN RED SOCIAL (REDDIT) ---
with col_paciente:
    st.subheader("📱 Interfaz del Usuario (Simulación Reddit)")
    st.info("El paciente escribe un post en su comunidad...")

    with st.form("reddit_form"):
        subreddit = st.selectbox("Subreddit", ["r/DeepThoughts", "r/Depression", "r/Teenagers", "r/Happy"])
        titulo = st.text_input("Título del Post")
        cuerpo = st.text_area("Contenido del Post (Body)", height=150)
        enviar = st.form_submit_button("Publicar Post")

# --- COLUMNA DERECHA: SISTEMA INTELIGENTE (MÉDICO/AGENTE) ---
with col_sistema:
    st.subheader("🛡️ Centro de Control del Agente")

    if enviar and cuerpo:
        with st.spinner('El Agente Recolector está procesando los datos...'):
            time.sleep(1) # Efecto visual de procesamiento

        # 1. Llamada al Modelo DL
        probabilidad, etiqueta = predecir_depresion_dl(cuerpo)

        # Mostrar Resultados Visuales
        st.write("### 1. Diagnóstico del Modelo (Deep Learning)")

        # Métrica grande
        delta_color = "inverse" if etiqueta == "Depresión" else "normal"
        st.metric(label="Clasificación del Modelo", value=etiqueta, delta=f"Riesgo: {probabilidad:.2%}", delta_color=delta_color)

        # Barra de progreso de riesgo
        st.write("Nivel de Riesgo Calculado:")
        color_barra = "red" if probabilidad > 0.5 else "green"
        st.progress(probabilidad)

        # 2. Llamada al Agente XAI
        st.write("### 2. Razonamiento del Super Agente (XAI)")
        with st.chat_message("assistant", avatar="🤖"):
            explicacion = agente_xai_explicacion(cuerpo, probabilidad)
            st.write(explicacion)

        # 3. Acción Sugerida (Super Agente)
        if probabilidad > 0.7:
            st.error("⚠️ ALERTA CRÍTICA: Se ha notificado al especialista humano.")
        elif probabilidad > 0.5:
            st.warning("⚠️ ALERTA: Se sugiere seguimiento preventivo.")
        else:
            st.success("✅ ESTADO: Sin riesgo aparente.")

    elif enviar and not cuerpo:
        st.warning("Por favor escribe algo en el contenido del post.")

    else:
        st.write("Esperando actividad del usuario...")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Reddit_logo_2007.svg/2560px-Reddit_logo_2007.svg.png", width=100, caption="Monitoreando r/Depression...")
