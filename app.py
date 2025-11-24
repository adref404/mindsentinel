"""
MindSentinel - Sistema Multi-Agente para Monitoreo de Salud Mental
===================================================================
Arquitectura: CrewAI + Google Gemini + Deep Learning (LSTM)
Frontend: Streamlit
Agentes:
    1. Clasificador (Deep Learning)
    2. Explicabilidad XAI (Gemini)
    3. Supervisor/Decisor Final (Gemini)
"""

import streamlit as st
import pickle
import numpy as np
import re
import os
from datetime import datetime
from dotenv import load_dotenv

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# CrewAI y LangChain
# ✅ CORRECCIÓN: Usar LLM de CrewAI directamente
from crewai import Agent, Task, Crew, Process, LLM

# ============================================================================
# CONFIGURACIÓN DE LA API DE GOOGLE GEMINI
# ============================================================================
# Cargar variables de entorno
load_dotenv()

# 🔑 IMPORTANTE: Configura tu API Key de Google Gemini aquí
# Obtén tu API Key gratis en: https://makersuite.google.com/app/apikey

# Opción 1: Variable de entorno (RECOMENDADO para producción)
# export GOOGLE_API_KEY='tu_api_key_aqui'
# Obtener API Key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Validar que la API Key esté configurada
if not GOOGLE_API_KEY:
    st.error("""
    ⚠️ **API Key no configurada**
    
    Por favor:
    1. Crea un archivo `.env` en la raíz del proyecto
    2. Agrega la línea: `GOOGLE_API_KEY=tu_api_key_aqui`
    3. Obtén tu API Key en: https://aistudio.google.com/app/apikey
    4. Reinicia la aplicación
    """)
    st.stop()

# ✅ SOLUCIÓN: Configurar LLM usando la clase LLM de CrewAI
try:
    llm = LLM(
        model="gemini/gemini-2.5-flash",  # Prefijo "gemini/" es OBLIGATORIO
        api_key=GOOGLE_API_KEY,
        temperature=0.7
    )
    
    # Test opcional: verificar que funciona
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    
except Exception as e:
    st.error(f"""
    ❌ **Error al configurar Google Gemini**
    
    Error: {str(e)}
    
    Posibles causas:
    - API Key inválida
    - Sin conexión a internet
    - Límite de uso excedido
    """)
    st.stop()
# ============================================================================
# CONFIGURACIÓN DE STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="MindSentinel - Monitoreo de Salud Mental",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .agent-card {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CARGA DE ARTEFACTOS DEL MODELO
# ============================================================================
@st.cache_resource
def load_artifacts():
    """
    Carga el modelo LSTM, tokenizador y configuración
    El decorador @st.cache_resource evita recargar en cada interacción
    """
    try:
        # Cargar modelo de Deep Learning
        model = load_model('modelo_depresion.h5')
        
        # Cargar tokenizador
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # Cargar configuración
        with open('model_config.pickle', 'rb') as handle:
            config = pickle.load(handle)
        
        return model, tokenizer, config
    
    except FileNotFoundError as e:
        st.error(f"""
        ❌ Error: No se encontraron los artefactos del modelo.
        
        Por favor asegúrate de:
        1. Ejecutar primero el script train_model.py
        2. Copiar los archivos generados a este directorio:
           - modelo_depresion.h5
           - tokenizer.pickle
           - model_config.pickle
        """)
        st.stop()

# Cargar artefactos
model, tokenizer, config = load_artifacts()

# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================================
def clean_text(text):
    """
    Limpieza de texto (IDÉNTICA a train_model.py)
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'r/\w+', '', text)
    text = re.sub(r'u/\w+', '', text)  # Agregar u/username
    text = re.sub(r'[^\w\s!?.\']', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_for_prediction(text, tokenizer, max_len):
    """
    Preprocesa texto para predicción del modelo
    """
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    return padded

# ============================================================================
# AGENTE 1: CLASIFICADOR (Deep Learning)
# ============================================================================
def agente_clasificador(text):
    """
    Agente 1: Clasificador de Depresión usando LSTM
    
    Returns:
        dict: {
            'probabilidad': float,
            'prediccion': str,
            'nivel_riesgo': str,
            'confianza': float
        }
    """
    # Preprocesar texto
    input_data = preprocess_for_prediction(text, tokenizer, config['max_len'])
    
    # Predicción
    probabilidad = float(model.predict(input_data, verbose=0)[0][0])
    
    # Determinar nivel de riesgo
    if probabilidad >= 0.7:
        nivel_riesgo = "ALTO"
        prediccion = "Indicadores significativos de depresión"
    elif probabilidad >= 0.4:
        nivel_riesgo = "MEDIO"
        prediccion = "Señales moderadas de riesgo"
    else:
        nivel_riesgo = "BAJO"
        prediccion = "Sin indicadores claros de depresión"
    
    return {
        'probabilidad': probabilidad,
        'prediccion': prediccion,
        'nivel_riesgo': nivel_riesgo,
        'confianza': abs(probabilidad - 0.5) * 2  # Normalizar confianza 0-1
    }

# ============================================================================
# AGENTE 2: EXPLICABILIDAD XAI (Google Gemini)
# ============================================================================
def crear_agente_explicabilidad():
    """
    Agente 2: Analista de Explicabilidad (XAI)
    Utiliza Gemini para explicar por qué el modelo hizo su predicción
    """
    agente = Agent(
        role='Psicólogo Computacional Especialista en XAI',
        goal='Explicar de forma clara y científica por qué el modelo detectó (o no) indicadores de depresión en el texto',
        backstory="""Eres un experto en Inteligencia Artificial Explicable (XAI) con maestría en psicología clínica.
        Tu trabajo es analizar texto y identificar:
        1. Palabras clave emocionales (negativas, desesperanza, aislamiento)
        2. Distorsiones cognitivas (pensamiento catastrófico, generalización excesiva)
        3. Patrones lingüísticos depresivos (primera persona, tiempo presente, absolutos)
        4. Tono emocional general (tristeza, anhedonia, desesperanza)
        
        Debes proporcionar explicaciones técnicas pero comprensibles, citando ejemplos específicos del texto.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    return agente

def tarea_explicar_prediccion(agente, texto_usuario, resultado_clasificador):
    """
    Tarea para el Agente de Explicabilidad
    """
    tarea = Task(
        description=f"""
        Analiza el siguiente texto de un post de Reddit y explica por qué el modelo de Deep Learning 
        predijo una probabilidad de {resultado_clasificador['probabilidad']:.2%} de depresión.
        
        TEXTO A ANALIZAR:
        "{texto_usuario}"
        
        PREDICCIÓN DEL MODELO:
        - Probabilidad de depresión: {resultado_clasificador['probabilidad']:.2%}
        - Nivel de riesgo: {resultado_clasificador['nivel_riesgo']}
        
        INSTRUCCIONES:
        1. Identifica palabras clave específicas del texto que indiquen estado emocional
        2. Detecta patrones lingüísticos asociados con depresión (uso de pronombres, tiempos verbales, absolutos)
        3. Identifica posibles distorsiones cognitivas (si las hay)
        4. Explica el tono emocional general
        5. Justifica por qué el modelo asignó esa probabilidad
        
        IMPORTANTE: Sé específico, cita fragmentos exactos del texto y mantén un tono profesional pero empático.
        """,
        expected_output="""Un análisis estructurado con las siguientes secciones:
        - **Palabras clave detectadas**: Lista de términos emocionales encontrados
        - **Patrones lingüísticos**: Análisis de estructura gramatical y uso del lenguaje
        - **Distorsiones cognitivas**: Identificación de sesgos en el pensamiento
        - **Tono emocional**: Descripción del estado anímico reflejado
        - **Justificación de la predicción**: Explicación coherente del score del modelo
        """,
        agent=agente
    )
    return tarea

# ============================================================================
# AGENTE 3: SUPERVISOR/DECISOR FINAL (Google Gemini)
# ============================================================================
def crear_agente_supervisor():
    """
    Agente 3: Supervisor Clínico
    Toma la decisión final y genera recomendaciones
    """
    agente = Agent(
        role='Supervisor Clínico de Salud Mental',
        goal='Tomar la decisión final sobre el nivel de intervención necesario y proporcionar recomendaciones apropiadas',
        backstory="""Eres un psiquiatra con 15 años de experiencia en salud mental digital.
        Tu responsabilidad es revisar los análisis del clasificador y del explicador, y decidir:
        
        - RIESGO ALTO (≥70%): Generar alerta clínica urgente con recomendaciones de intervención inmediata
        - RIESGO MEDIO (40-69%): Sugerir monitoreo cercano y recursos de apoyo
        - RIESGO BAJO (<40%): Proporcionar mensaje de refuerzo positivo y recursos preventivos
        
        Siempre debes ser empático, profesional y proporcionar recursos concretos (líneas de ayuda, terapias, apps).""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    return agente

def tarea_decision_final(agente, texto_usuario, resultado_clasificador, explicacion_xai):
    """
    Tarea para el Agente Supervisor
    """
    tarea = Task(
        description=f"""
        Como Supervisor Clínico, revisa el caso completo y proporciona tu decisión final.
        
        INFORMACIÓN DEL CASO:
        
        Texto del usuario:
        "{texto_usuario}"
        
        Resultado del Clasificador (LSTM):
        - Probabilidad de depresión: {resultado_clasificador['probabilidad']:.2%}
        - Nivel de riesgo: {resultado_clasificador['nivel_riesgo']}
        - Confianza del modelo: {resultado_clasificador['confianza']:.2%}
        
        Análisis de Explicabilidad (XAI):
        {explicacion_xai}
        
        TU TAREA:
        1. Evalúa la coherencia entre la predicción del modelo y el análisis XAI
        2. Determina el nivel de intervención requerido:
           - ALERTA CLÍNICA URGENTE (riesgo alto)
           - MONITOREO Y APOYO (riesgo medio)
           - REFUERZO POSITIVO (riesgo bajo)
        
        3. Proporciona recomendaciones específicas:
           - Líneas de ayuda (España: 024, México: 800 290 0024, etc.)
           - Tipos de terapia recomendados
           - Recursos digitales (apps, comunidades de apoyo)
           - Acciones inmediatas a tomar
        
        4. Redacta un mensaje final para el usuario (empático pero profesional)
        """,
        expected_output="""Un informe de supervisión con:
        - **Decisión clínica**: Nivel de intervención determinado
        - **Justificación**: Por qué se tomó esa decisión
        - **Recomendaciones específicas**: Lista de recursos y acciones
        - **Mensaje para el usuario**: Comunicación empática y orientadora
        - **Próximos pasos**: Qué debe hacer el usuario de inmediato
        """,
        agent=agente
    )
    return tarea

# ============================================================================
# FUNCIÓN PRINCIPAL: ORQUESTACIÓN DE AGENTES CON CREWAI
# ============================================================================
def ejecutar_sistema_multiagente(titulo, cuerpo):
    """
    Orquesta los 3 agentes para analizar el texto del usuario
    
    Flujo:
    1. Agente Clasificador → Predicción LSTM
    2. Agente XAI → Explicación de la predicción
    3. Agente Supervisor → Decisión final y recomendaciones
    """
    
    # Combinar título y cuerpo
    texto_completo = f"{titulo}. {cuerpo}"
    
    # ========== AGENTE 1: CLASIFICADOR ==========
    with st.spinner("🔍 Agente 1: Analizando con modelo LSTM..."):
        resultado_clasificador = agente_clasificador(texto_completo)
    
    st.success(f"✅ Clasificador completado: {resultado_clasificador['prediccion']}")
    
    # ========== AGENTE 2: EXPLICABILIDAD ==========
    with st.spinner("🧠 Agente 2: Generando explicación XAI con Gemini..."):
        agente_xai = crear_agente_explicabilidad()
        tarea_xai = tarea_explicar_prediccion(agente_xai, texto_completo, resultado_clasificador)
        
        crew_xai = Crew(
            agents=[agente_xai],
            tasks=[tarea_xai],
            process=Process.sequential,
            verbose=True
        )
        
        resultado_xai = crew_xai.kickoff()
        explicacion_xai = resultado_xai.raw if hasattr(resultado_xai, 'raw') else str(resultado_xai)
    
    st.success("✅ Explicabilidad completada")
    
    # ========== AGENTE 3: SUPERVISOR ==========
    with st.spinner("👨‍⚕️ Agente 3: Supervisor generando recomendaciones..."):
        agente_supervisor = crear_agente_supervisor()
        tarea_supervisor = tarea_decision_final(
            agente_supervisor, 
            texto_completo, 
            resultado_clasificador, 
            explicacion_xai
        )
        
        crew_supervisor = Crew(
            agents=[agente_supervisor],
            tasks=[tarea_supervisor],
            process=Process.sequential,
            verbose=True
        )
        
        resultado_supervisor = crew_supervisor.kickoff()
        decision_final = resultado_supervisor.raw if hasattr(resultado_supervisor, 'raw') else str(resultado_supervisor)
    
    st.success("✅ Supervisión completada")
    
    return {
        'clasificador': resultado_clasificador,
        'explicacion': explicacion_xai,
        'decision': decision_final
    }

# ============================================================================
# INTERFAZ DE USUARIO PRINCIPAL
# ============================================================================

# Header
st.markdown('<p class="main-header">🧠 MindSentinel</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema Multi-Agente para Monitoreo de Salud Mental en Redes Sociales</p>', unsafe_allow_html=True)

# Sidebar con información
with st.sidebar:
    st.image("https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Brain/3D/brain_3d.png", width=100)
    st.title("Información del Sistema")
    
    st.markdown("""
    ### 🏗️ Arquitectura
    **3 Agentes Especializados:**
    
    1. 🤖 **Clasificador** (LSTM Bidireccional)
       - Analiza el texto con Deep Learning
       - Genera probabilidad de depresión
    
    2. 🧠 **Explicador XAI** (Gemini Flash)
       - Explica la predicción del modelo
       - Identifica patrones lingüísticos
    
    3. 👨‍⚕️ **Supervisor Clínico** (Gemini Flash)
       - Toma decisión final
       - Genera recomendaciones
    
    ### 📊 Métricas del Modelo
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{config['test_accuracy']:.2%}")
        st.metric("Precision", f"{config['test_precision']:.2%}")
    with col2:
        st.metric("AUC-ROC", f"{config['test_auc']:.2%}")
        st.metric("Recall", f"{config['test_recall']:.2%}")
    
    st.markdown("---")
    st.markdown("""
    ### ⚠️ Aviso Importante
    Este sistema es una herramienta de **apoyo académico**.
    NO reemplaza el diagnóstico profesional.
    
    **En crisis, contacta:**
    - 🇪🇸 España: 024
    - 🇲🇽 México: 800 290 0024
    - 🇦🇷 Argentina: 135
    """)

# Área principal de entrada
st.markdown("## 📝 Simula un Post de Reddit")

col1, col2 = st.columns([1, 1])

with col1:
    titulo = st.text_input(
        "Título del Post",
        placeholder="Ej: No sé qué hacer con mi vida...",
        help="Escribe el título como aparecería en Reddit"
    )

with col2:
    subreddit = st.selectbox(
        "Subreddit",
        ["r/depression", "r/mentalhealth", "r/anxiety", "r/therapy", "r/offmychest"],
        help="Contexto del subreddit (informativo)"
    )

cuerpo = st.text_area(
    "Cuerpo del Post (Body)",
    placeholder="""Escribe aquí el contenido del post...

Ejemplo:
Últimamente me siento completamente vacío. No encuentro motivación para hacer nada, ni siquiera las cosas que antes me gustaban. Siento que soy una carga para todos y que nada tiene sentido. No sé si esto va a mejorar algún día...""",
    height=200,
    help="Contenido principal del post que será analizado"
)

# Botón de análisis
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analizar_btn = st.button("🔍 Analizar con MindSentinel", type="primary", use_container_width=True)

# Procesamiento
if analizar_btn:
    if not titulo or not cuerpo:
        st.error("⚠️ Por favor completa tanto el título como el cuerpo del post")
    elif len(cuerpo) < 20:
        st.warning("⚠️ El texto es muy corto. Escribe al menos 20 caracteres para un análisis preciso.")
    else:
        # Mostrar el post simulado
        with st.expander("📄 Post a analizar", expanded=True):
            st.markdown(f"### {titulo}")
            st.markdown(f"*Publicado en {subreddit} • {datetime.now().strftime('%d/%m/%Y %H:%M')}*")
            st.markdown(f"{cuerpo}")
        
        st.markdown("---")
        st.markdown("## 🤖 Análisis del Sistema Multi-Agente")
        
        # Ejecutar sistema multi-agente
        try:
            resultados = ejecutar_sistema_multiagente(titulo, cuerpo)
            
            # ========== RESULTADOS DEL CLASIFICADOR ==========
            st.markdown("### 1️⃣ Agente Clasificador (Deep Learning)")
            
            prob = resultados['clasificador']['probabilidad']
            nivel = resultados['clasificador']['nivel_riesgo']
            
            # Determinar clase CSS
            if nivel == "ALTO":
                css_class = "high-risk"
                emoji = "🔴"
            elif nivel == "MEDIO":
                css_class = "medium-risk"
                emoji = "🟡"
            else:
                css_class = "low-risk"
                emoji = "🟢"
            
            st.markdown(f"""
            <div class="prediction-box {css_class}">
                <h2>{emoji} Nivel de Riesgo: {nivel}</h2>
                <p style="font-size: 1.5rem; margin: 10px 0;">
                    <strong>Probabilidad de Depresión: {prob:.1%}</strong>
                </p>
                <p>{resultados['clasificador']['prediccion']}</p>
                <p><em>Confianza del modelo: {resultados['clasificador']['confianza']:.1%}</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Barra de progreso visual
            st.progress(prob)
            
            # ========== EXPLICACIÓN XAI ==========
            st.markdown("### 2️⃣ Agente Explicador (XAI con Gemini)")
            with st.container():
                st.markdown(f"""
                <div class="agent-card">
                {resultados['explicacion']}
                </div>
                """, unsafe_allow_html=True)
            
            # ========== DECISIÓN FINAL ==========
            st.markdown("### 3️⃣ Agente Supervisor (Decisión Clínica)")
            with st.container():
                st.markdown(f"""
                <div class="agent-card">
                {resultados['decision']}
                </div>
                """, unsafe_allow_html=True)
            
            # ========== RECURSOS ADICIONALES ==========
            st.markdown("---")
            st.markdown("## 📞 Recursos de Ayuda Inmediata")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("""
                **🇪🇸 España**
                - Teléfono: 024
                - Servicio 24/7 gratuito
                """)
            
            with col2:
                st.info("""
                **🇲🇽 México**
                - Teléfono: 800 290 0024
                - SAPTEL 24 horas
                """)
            
            with col3:
                st.info("""
                **🌍 Internacional**
                - findahelpline.com
                - Recursos por país
                """)
            
        except Exception as e:
            st.error(f"❌ Error durante el análisis: {str(e)}")
            st.info("Verifica que la GOOGLE_API_KEY esté correctamente configurada")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>MindSentinel</strong> • Sistema Multi-Agente para Salud Mental</p>
    <p>Desarrollado con ❤️ usando TensorFlow, CrewAI y Google Gemini</p>
    <p><em>⚠️ Herramienta académica - No sustituye atención médica profesional</em></p>
</div>
""", unsafe_allow_html=True)
