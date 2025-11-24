"""
MindSentinel - Sistema Multi-Agente para Monitoreo de Salud Mental
===================================================================
Versión Profesional con Base de Datos y Autenticación de Pacientes

Arquitectura: CrewAI + Google Gemini + Deep Learning (LSTM) + SQLite
Frontend: Streamlit con tema clínico profesional
"""

import streamlit as st
import pickle
import numpy as np
import re
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# CrewAI y LLM
from crewai import Agent, Task, Crew, Process, LLM

# Base de datos
import sys

sys.path.append(os.path.dirname(__file__))
from database.db_manager import DatabaseManager

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

# Cargar variables de entorno
load_dotenv()

# API Key de Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error(
        "ERROR: GOOGLE_API_KEY no configurada. Configura tu API Key en el archivo .env"
    )
    st.stop()

# Configurar LLM
try:
    llm = LLM(model="gemini/gemini-2.5-flash", api_key=GOOGLE_API_KEY, temperature=0.7)
except Exception as e:
    st.error(f"Error al configurar Google Gemini: {str(e)}")
    st.stop()

# Inicializar base de datos
db = DatabaseManager()
db.initialize_database()

# ============================================================================
# CONFIGURACIÓN DE STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="MindSentinel - Sistema Clínico de Salud Mental",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)



# Cargar CSS personalizado
def load_css():
    """Cargar CSS profesional"""
    css_path = os.path.join(
        os.path.dirname(__file__), "static", "css", "clinical_theme.css"
    )

    if os.path.exists(css_path):
        # 👈 AQUI SE AGREGA encoding="utf-8"
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # CSS mínimo si no encuentra el archivo
        st.markdown(
            """
        <style>
        :root {
            --primary-blue: #0056b3;
            --risk-critical: #dc3545;
            --risk-warning: #fd7e14;
            --risk-safe: #28a745;
        }
        .clinical-header {
            background: linear-gradient(135deg, #0056b3 0%, #003d82 100%);
            color: white;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
            border-radius: 8px;
        }
        .clinical-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        .risk-assessment {
            padding: 2rem;
            border-radius: 8px;
            border-left: 5px solid;
            margin: 1rem 0;
        }
        .high-risk { border-left-color: #dc3545; background: #fff5f5; }
        .medium-risk { border-left-color: #fd7e14; background: #fff8f0; }
        .low-risk { border-left-color: #28a745; background: #f0fff4; }
        </style>
        """,
            unsafe_allow_html=True,
        )


load_css()

# ============================================================================
# GESTIÓN DE SESIONES
# ============================================================================

# Inicializar estado de sesión
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.paciente_id = None
    st.session_state.paciente_info = None
    st.session_state.page = "login"

# ============================================================================
# CARGA DE ARTEFACTOS DEL MODELO
# ============================================================================


@st.cache_resource
def load_artifacts():
    """Cargar modelo LSTM, tokenizador y configuración"""
    try:
        model = load_model("modelo_depresion.h5")

        with open("tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)

        with open("model_config.pickle", "rb") as handle:
            config = pickle.load(handle)

        return model, tokenizer, config

    except FileNotFoundError:
        st.error(
            """
        ERROR: No se encontraron los artefactos del modelo.
        
        Ejecuta primero: python train_model.py
        
        Archivos requeridos:
        - modelo_depresion.h5
        - tokenizer.pickle
        - model_config.pickle
        """
        )
        st.stop()


model, tokenizer, config = load_artifacts()

# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================================


def clean_text(text):
    """Limpieza de texto (idéntica a train_model.py)"""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"r/\w+", "", text)
    text = re.sub(r"u/\w+", "", text)
    text = re.sub(r"[^\w\s!?.\']", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_for_prediction(text, tokenizer, max_len):
    """Preprocesar texto para predicción"""
    cleaned = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len, padding="post", truncating="post")
    return padded


# ============================================================================
# AGENTE 1: CLASIFICADOR (Deep Learning)
# ============================================================================


def agente_clasificador(text):
    """Agente 1: Clasificador LSTM"""
    input_data = preprocess_for_prediction(text, tokenizer, config["max_len"])
    probabilidad = float(model.predict(input_data, verbose=0)[0][0])

    if probabilidad >= 0.7:
        nivel_riesgo = "ALTO"
        prediccion = "Indicadores significativos de depresión detectados"
    elif probabilidad >= 0.4:
        nivel_riesgo = "MEDIO"
        prediccion = "Señales moderadas de riesgo identificadas"
    else:
        nivel_riesgo = "BAJO"
        prediccion = "Sin indicadores claros de depresión"

    return {
        "probabilidad": probabilidad,
        "prediccion": prediccion,
        "nivel_riesgo": nivel_riesgo,
        "confianza": abs(probabilidad - 0.5) * 2,
    }


# ============================================================================
# AGENTE 2: EXPLICABILIDAD XAI (Google Gemini)
# ============================================================================


def crear_agente_explicabilidad():
    """Agente 2: Analista de Explicabilidad (XAI)"""
    agente = Agent(
        role="Psicólogo Computacional Especialista en XAI",
        goal="Explicar científicamente por qué el modelo detectó indicadores de depresión",
        backstory="""Experto en Inteligencia Artificial Explicable (XAI) con maestría en psicología clínica.
        Analizas texto identificando palabras clave emocionales, distorsiones cognitivas, 
        patrones lingüísticos depresivos y tono emocional. Proporcionas explicaciones técnicas 
        pero comprensibles, citando ejemplos específicos del texto.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    return agente


def tarea_explicar_prediccion(agente, texto_usuario, resultado_clasificador):
    """Tarea para el Agente de Explicabilidad"""
    tarea = Task(
        description=f"""
        Analiza el siguiente texto y explica por qué el modelo predijo {resultado_clasificador['probabilidad']:.2%} 
        de probabilidad de depresión.
        
        TEXTO: "{texto_usuario}"
        
        PREDICCIÓN: {resultado_clasificador['probabilidad']:.2%} - {resultado_clasificador['nivel_riesgo']}
        
        ANÁLISIS REQUERIDO:
        1. Palabras clave específicas que indiquen estado emocional
        2. Patrones lingüísticos asociados con depresión
        3. Posibles distorsiones cognitivas
        4. Tono emocional general
        5. Justificación de la probabilidad asignada
        
        Sé específico, cita fragmentos del texto y mantén tono profesional.
        """,
        expected_output="""Análisis estructurado con:
        - Palabras clave detectadas
        - Patrones lingüísticos
        - Distorsiones cognitivas
        - Tono emocional
        - Justificación de la predicción
        """,
        agent=agente,
    )
    return tarea


# ============================================================================
# AGENTE 3: SUPERVISOR CLÍNICO (Google Gemini)
# ============================================================================


def crear_agente_supervisor():
    """Agente 3: Supervisor Clínico"""
    agente = Agent(
        role="Supervisor Clínico de Salud Mental",
        goal="Tomar decisión final sobre el nivel de intervención necesario",
        backstory="""Psiquiatra con 15 años de experiencia en salud mental digital.
        Revisas análisis del clasificador y explicador, y decides:
        - RIESGO ALTO (≥70%): Alerta clínica urgente con intervención inmediata
        - RIESGO MEDIO (40-69%): Monitoreo cercano y recursos de apoyo
        - RIESGO BAJO (<40%): Refuerzo positivo y recursos preventivos
        
        Siempre empático, profesional y con recursos concretos.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
    return agente


def tarea_decision_final(
    agente, texto_usuario, resultado_clasificador, explicacion_xai
):
    """Tarea para el Agente Supervisor"""
    tarea = Task(
        description=f"""
        Como Supervisor Clínico, revisa el caso completo:
        
        TEXTO: "{texto_usuario}"
        
        CLASIFICADOR:
        - Probabilidad: {resultado_clasificador['probabilidad']:.2%}
        - Nivel de riesgo: {resultado_clasificador['nivel_riesgo']}
        - Confianza: {resultado_clasificador['confianza']:.2%}
        
        ANÁLISIS XAI:
        {explicacion_xai}
        
        TAREAS:
        1. Evaluar coherencia entre predicción y análisis XAI
        2. Determinar nivel de intervención (Alerta Urgente / Monitoreo / Refuerzo Positivo)
        3. Proporcionar recomendaciones específicas (líneas de ayuda, terapias, recursos digitales)
        4. Redactar mensaje final para el paciente (empático pero profesional)
        """,
        expected_output="""Informe de supervisión con:
        - Decisión clínica
        - Justificación
        - Recomendaciones específicas
        - Mensaje para el paciente
        - Próximos pasos
        """,
        agent=agente,
    )
    return tarea


# ============================================================================
# FUNCIÓN PRINCIPAL: ORQUESTACIÓN DE AGENTES
# ============================================================================


def ejecutar_sistema_multiagente(titulo, cuerpo, paciente_id):
    """Orquesta los 3 agentes y guarda resultados en BD"""

    texto_completo = f"{titulo}. {cuerpo}"
    inicio_tiempo = time.time()

    # AGENTE 1: Clasificador
    with st.spinner("Agente Clasificador analizando con LSTM..."):
        resultado_clasificador = agente_clasificador(texto_completo)

    st.success("Clasificador completado")

    # AGENTE 2: Explicabilidad
    with st.spinner("Agente Explicador generando análisis XAI..."):
        agente_xai = crear_agente_explicabilidad()
        tarea_xai = tarea_explicar_prediccion(
            agente_xai, texto_completo, resultado_clasificador
        )

        crew_xai = Crew(
            agents=[agente_xai],
            tasks=[tarea_xai],
            process=Process.sequential,
            verbose=True,
        )

        resultado_xai = crew_xai.kickoff()
        explicacion_xai = (
            resultado_xai.raw if hasattr(resultado_xai, "raw") else str(resultado_xai)
        )

    st.success("Explicabilidad completada")

    # AGENTE 3: Supervisor
    with st.spinner("Agente Supervisor generando recomendaciones..."):
        agente_supervisor = crear_agente_supervisor()
        tarea_supervisor = tarea_decision_final(
            agente_supervisor, texto_completo, resultado_clasificador, explicacion_xai
        )

        crew_supervisor = Crew(
            agents=[agente_supervisor],
            tasks=[tarea_supervisor],
            process=Process.sequential,
            verbose=True,
        )

        resultado_supervisor = crew_supervisor.kickoff()
        decision_final = (
            resultado_supervisor.raw
            if hasattr(resultado_supervisor, "raw")
            else str(resultado_supervisor)
        )

    st.success("Supervisión completada")

    duracion = time.time() - inicio_tiempo

    # Guardar en base de datos
    datos_evaluacion = {
        "titulo_post": titulo,
        "cuerpo_post": cuerpo,
        "subreddit": st.session_state.get("subreddit_seleccionado", "N/A"),
        "probabilidad_depresion": resultado_clasificador["probabilidad"],
        "nivel_riesgo": resultado_clasificador["nivel_riesgo"],
        "confianza_modelo": resultado_clasificador["confianza"],
        "prediccion_texto": resultado_clasificador["prediccion"],
        "analisis_xai": explicacion_xai,
        "decision_supervisor": decision_final,
        "recomendaciones": decision_final,  # Extraer recomendaciones del texto
        "nivel_intervencion": resultado_clasificador["nivel_riesgo"],
        "duracion_analisis_segundos": duracion,
    }

    evaluacion_id = db.guardar_evaluacion(paciente_id, datos_evaluacion)

    return {
        "clasificador": resultado_clasificador,
        "explicacion": explicacion_xai,
        "decision": decision_final,
        "evaluacion_id": evaluacion_id,
        "duracion": duracion,
    }


# ============================================================================
# PÁGINAS DE LA APLICACIÓN
# ============================================================================


def pagina_login():
    """Página de inicio de sesión"""
    st.markdown(
        """
    <div class="clinical-header">
        <div class="logo-icon">⚕️</div>
        <h1>MindSentinel</h1>
        <div class="subtitle">Sistema Multi-Agente para Monitoreo de Salud Mental</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown('<div class="clinical-card">', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Iniciar Sesión", "Registrarse"])

        # TAB 1: Login
        with tab1:
            st.markdown("### Acceso de Pacientes")

            email = st.text_input(
                "Correo Electrónico",
                key="login_email",
                placeholder="paciente@example.com",
            )
            password = st.text_input(
                "Contraseña", type="password", key="login_password"
            )

            if st.button("Iniciar Sesión", type="primary", use_container_width=True):
                if email and password:
                    paciente = db.autenticar_paciente(email, password)

                    if paciente:
                        st.session_state.authenticated = True
                        st.session_state.paciente_id = paciente["id"]
                        st.session_state.paciente_info = paciente
                        st.session_state.page = "dashboard"
                        st.success(f"Bienvenido, {paciente['nombre_completo']}")
                        st.rerun()
                    else:
                        st.error("Credenciales incorrectas")
                else:
                    st.warning("Por favor completa todos los campos")

        # TAB 2: Registro
        with tab2:
            st.markdown("### Registro de Nuevo Paciente")

            nombre = st.text_input(
                "Nombre Completo", key="reg_nombre", placeholder="Ej: Juan Pérez"
            )
            edad = st.number_input(
                "Edad", min_value=13, max_value=100, value=25, key="reg_edad"
            )
            genero = st.selectbox(
                "Género",
                ["Masculino", "Femenino", "Otro", "Prefiero no decir"],
                key="reg_genero",
            )
            email_reg = st.text_input(
                "Correo Electrónico", key="reg_email", placeholder="tu@email.com"
            )
            telefono = st.text_input(
                "Teléfono (opcional)", key="reg_telefono", placeholder="+51 999 999 999"
            )
            password_reg = st.text_input(
                "Contraseña", type="password", key="reg_password"
            )
            password_confirm = st.text_input(
                "Confirmar Contraseña", type="password", key="reg_password_confirm"
            )

            if st.button("Registrarse", type="primary", use_container_width=True):
                if nombre and edad and email_reg and password_reg:
                    if password_reg == password_confirm:
                        paciente = db.registrar_paciente(
                            nombre, edad, genero, email_reg, password_reg, telefono
                        )

                        if paciente:
                            st.success(
                                f"Registro exitoso! Tu código de paciente es: {paciente['codigo_paciente']}"
                            )
                            st.info(
                                "Ahora puedes iniciar sesión con tu correo y contraseña"
                            )
                        else:
                            st.error("Error al registrar. El correo ya está en uso.")
                    else:
                        st.error("Las contraseñas no coinciden")
                else:
                    st.warning("Por favor completa todos los campos obligatorios")

        st.markdown("</div>", unsafe_allow_html=True)


def pagina_dashboard():
    """Dashboard principal del paciente"""
    paciente = st.session_state.paciente_info

    # Header con información del paciente
    st.markdown(
        f"""
    <div class="clinical-header">
        <h1>MindSentinel - Panel del Paciente</h1>
        <div class="subtitle">
            Paciente: {paciente['nombre_completo']} | Código: {paciente['codigo_paciente']}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Botón de cerrar sesión
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Cerrar Sesión"):
            st.session_state.authenticated = False
            st.session_state.paciente_id = None
            st.session_state.paciente_info = None
            st.session_state.page = "login"
            st.rerun()

    # Tabs principales
    tab1, tab2, tab3 = st.tabs(
        ["Nueva Evaluación", "Historial Clínico", "Estadísticas"]
    )

    with tab1:
        pagina_evaluacion()

    with tab2:
        pagina_historial()

    with tab3:
        pagina_estadisticas()


def pagina_evaluacion():
    """Página de nueva evaluación"""
    st.markdown("## Simulación de Post de Reddit")
    st.markdown(
        "Por favor, completa los siguientes campos como si fueras a publicar en Reddit:"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        titulo = st.text_input(
            "Título del Post",
            placeholder="Ej: No sé qué hacer con mi vida...",
            help="Escribe el título como aparecería en Reddit",
        )

    with col2:
        subreddit = st.selectbox(
            "Subreddit",
            [
                "r/depression",
                "r/mentalhealth",
                "r/anxiety",
                "r/therapy",
                "r/offmychest",
            ],
            help="Contexto del subreddit",
        )
        st.session_state.subreddit_seleccionado = subreddit

    cuerpo = st.text_area(
        "Cuerpo del Post",
        placeholder="""Escribe aquí el contenido del post...

Ejemplo:
Últimamente me siento completamente vacío. No encuentro motivación para hacer nada...""",
        height=200,
        help="Contenido principal del post que será analizado",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analizar_btn = st.button(
            "Analizar con MindSentinel", type="primary", use_container_width=True
        )

    if analizar_btn:
        if not titulo or not cuerpo:
            st.error("Por favor completa tanto el título como el cuerpo del post")
        elif len(cuerpo) < 20:
            st.warning(
                "El texto es muy corto. Escribe al menos 20 caracteres para un análisis preciso."
            )
        else:
            # Mostrar post a analizar
            with st.expander("Post a analizar", expanded=True):
                st.markdown(f"### {titulo}")
                st.markdown(
                    f"*Publicado en {subreddit} • {datetime.now().strftime('%d/%m/%Y %H:%M')}*"
                )
                st.markdown(f"{cuerpo}")

            st.markdown("---")
            st.markdown("## Análisis del Sistema Multi-Agente")

            # Ejecutar sistema
            try:
                resultados = ejecutar_sistema_multiagente(
                    titulo, cuerpo, st.session_state.paciente_id
                )

                # Mostrar resultados
                mostrar_resultados_evaluacion(resultados)

                st.success(f"Evaluación guardada con ID: {resultados['evaluacion_id']}")
                st.info("Puedes ver esta evaluación en tu Historial Clínico")

            except Exception as e:
                st.error(f"Error durante el análisis: {str(e)}")


def mostrar_resultados_evaluacion(resultados):
    """Mostrar resultados de la evaluación"""

    # RESULTADO DEL CLASIFICADOR
    st.markdown("### Agente 1: Clasificador (Deep Learning)")

    prob = resultados["clasificador"]["probabilidad"]
    nivel = resultados["clasificador"]["nivel_riesgo"]

    if nivel == "ALTO":
        css_class = "high-risk"
        badge = "badge-critical"
    elif nivel == "MEDIO":
        css_class = "medium-risk"
        badge = "badge-warning"
    else:
        css_class = "low-risk"
        badge = "badge-safe"

    st.markdown(
        f"""
    <div class="risk-assessment {css_class}">
        <h2>Nivel de Riesgo: <span class="clinical-badge {badge}">{nivel}</span></h2>
        <div class="probability-display">{prob:.1%}</div>
        <p style="font-size: 1.1rem;">{resultados['clasificador']['prediccion']}</p>
        <p><em>Confianza del modelo: {resultados['clasificador']['confianza']:.1%}</em></p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Progress bar
    st.progress(prob)

    # EXPLICACIÓN XAI
    st.markdown("### Agente 2: Explicador (XAI con Gemini)")
    st.markdown(
        f"""
    <div class="agent-section">
        <div class="agent-content">{resultados['explicacion']}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # DECISIÓN SUPERVISOR
    st.markdown("### Agente 3: Supervisor (Decisión Clínica)")
    st.markdown(
        f"""
    <div class="agent-section">
        <div class="agent-content">{resultados['decision']}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # RECURSOS DE AYUDA
    st.markdown("---")
    st.markdown("## Recursos de Ayuda Inmediata")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            """
        **Perú**  
        Teléfono: 0800-10828  
        Servicio 24/7 gratuito
        """
        )

    with col2:
        st.info(
            """
        **México**  
        Teléfono: 800 290 0024  
        SAPTEL 24 horas
        """
        )

    with col3:
        st.info(
            """
        **Internacional**  
        findahelpline.com  
        Recursos por país
        """
        )


def pagina_historial():
    """Página de historial clínico"""
    st.markdown("## Historial de Evaluaciones")

    paciente_id = st.session_state.paciente_id
    evaluaciones = db.obtener_evaluaciones_paciente(paciente_id, limite=50)

    if not evaluaciones:
        st.info(
            "No hay evaluaciones registradas. Realiza tu primera evaluación en la pestaña 'Nueva Evaluación'."
        )
        return

    st.markdown(f"**Total de evaluaciones:** {len(evaluaciones)}")

    # Mostrar evaluaciones en tabla
    for eval in evaluaciones:
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

        with col1:
            st.markdown(f"**{eval['titulo_post']}**")
            st.caption(eval["fecha_evaluacion"])

        with col2:
            st.metric("Probabilidad", f"{eval['probabilidad_depresion']:.1%}")

        with col3:
            nivel = eval["nivel_riesgo"]
            if nivel == "ALTO":
                st.markdown(
                    f"<span class='clinical-badge badge-critical'>{nivel}</span>",
                    unsafe_allow_html=True,
                )
            elif nivel == "MEDIO":
                st.markdown(
                    f"<span class='clinical-badge badge-warning'>{nivel}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<span class='clinical-badge badge-safe'>{nivel}</span>",
                    unsafe_allow_html=True,
                )

        with col4:
            if st.button("Ver", key=f"ver_{eval['id']}"):
                st.session_state.evaluacion_ver_id = eval["id"]
                st.rerun()

        st.markdown("---")

    # Mostrar detalles si se seleccionó una evaluación
    if "evaluacion_ver_id" in st.session_state:
        mostrar_detalle_evaluacion(st.session_state.evaluacion_ver_id)


def mostrar_detalle_evaluacion(evaluacion_id):
    """Mostrar detalles completos de una evaluación"""
    eval_completa = db.obtener_evaluacion_completa(evaluacion_id)

    if eval_completa:
        st.markdown("## Detalle de Evaluación")

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Volver al historial"):
                del st.session_state.evaluacion_ver_id
                st.rerun()

        st.markdown(f"**Fecha:** {eval_completa['fecha_evaluacion']}")
        st.markdown(f"**Título:** {eval_completa['titulo_post']}")
        st.markdown(f"**Subreddit:** {eval_completa['subreddit']}")

        st.markdown("### Texto Analizado")
        st.text_area("", eval_completa["cuerpo_post"], height=150, disabled=True)

        st.markdown("### Resultados del Análisis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Probabilidad Depresión",
                f"{eval_completa['probabilidad_depresion']:.1%}",
            )
        with col2:
            st.metric("Nivel de Riesgo", eval_completa["nivel_riesgo"])
        with col3:
            st.metric("Confianza Modelo", f"{eval_completa['confianza_modelo']:.1%}")

        st.markdown("### Análisis XAI")
        st.markdown(eval_completa["analisis_xai"])

        st.markdown("### Decisión del Supervisor")
        st.markdown(eval_completa["decision_supervisor"])


def pagina_estadisticas():
    """Página de estadísticas del paciente"""
    st.markdown("## Estadísticas de tu Historial")

    paciente_id = st.session_state.paciente_id
    stats = db.obtener_estadisticas_paciente(paciente_id)

    if stats and stats["total_evaluaciones"] > 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{stats["total_evaluaciones"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="metric-label">Total Evaluaciones</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{stats["promedio_probabilidad"]:.1%}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="metric-label">Promedio Probabilidad</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{stats["maxima_probabilidad"]:.1%}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="metric-label">Máxima Detectada</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-value">{stats["minima_probabilidad"]:.1%}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="metric-label">Mínima Detectada</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### Distribución de Evaluaciones por Nivel de Riesgo")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Riesgo Alto",
                stats["evaluaciones_alto_riesgo"],
                delta=None,
                delta_color="inverse",
            )

        with col2:
            st.metric("Riesgo Medio", stats["evaluaciones_medio_riesgo"])

        with col3:
            st.metric(
                "Riesgo Bajo",
                stats["evaluaciones_bajo_riesgo"],
                delta=None,
                delta_color="normal",
            )

    else:
        st.info(
            "No hay suficientes datos para mostrar estadísticas. Realiza al menos una evaluación."
        )


# ============================================================================
# LÓGICA PRINCIPAL DE NAVEGACIÓN
# ============================================================================


def main():
    """Función principal de la aplicación"""

    if not st.session_state.authenticated:
        pagina_login()
    else:
        pagina_dashboard()

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div class="clinical-footer">
        <p><strong>MindSentinel</strong> - Sistema Multi-Agente para Salud Mental</p>
        <p>Desarrollado con TensorFlow, CrewAI y Google Gemini</p>
        <p><em>Herramienta académica - No sustituye atención médica profesional</em></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
