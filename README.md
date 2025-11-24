# 🧠 MindSentinel - Sistema Multi-Agente para Monitoreo de Salud Mental

## 📖 Descripción del Proyecto

**MindSentinel** es un sistema académico de inteligencia artificial que utiliza arquitectura multi-agente para detectar indicadores de depresión en texto simulado de redes sociales (Reddit). Combina:

- **Deep Learning (LSTM Bidireccional)** para clasificación de texto
- **CrewAI** para orquestación de agentes inteligentes
- **Google Gemini (Flash)** para explicabilidad e interpretación clínica
- **Streamlit** para interfaz de usuario interactiva

---

## 📋 Información General

| Campo | Detalle |
|-------|---------|
| **Tipo** | Sistema Multi-Agente para Monitoreo de Salud Mental |
| **Objetivo** | Detectar indicadores de depresión en texto de redes sociales (Reddit) |
| **Nivel** | Proyecto Académico / Investigación |
| **Tecnologías Core** | TensorFlow, CrewAI, Google Gemini, Streamlit |
| **Versión** | 1.0 (Noviembre 2025) |
| **Contacto** | fernando.celadita@unmsm.edu.pe |

---

## 🎯 Características Principales

### ✅ Análisis Automático con Deep Learning
- Modelo LSTM Bidireccional entrenado en 7,500+ posts de Reddit
- Accuracy: ~85-90%
- Detección de patrones lingüísticos asociados a depresión

### ✅ Sistema Multi-Agente Inteligente
- **Agente 1 (Clasificador)**: Predicción con Deep Learning
- **Agente 2 (Explicador XAI)**: Interpretabilidad con Gemini
- **Agente 3 (Supervisor)**: Decisión clínica y recomendaciones

### ✅ Interfaz de Usuario Amigable
- Streamlit para UI interactiva
- Visualización clara de resultados
- Recursos de ayuda integrados

---

## 📦 Archivos del Proyecto

```
mindsentinel/
│
├── train_model.py              # Script de entrenamiento (16 KB) - Parte 1
├── app.py                      # Aplicación multi-agente (23 KB) - Parte 2
├── requirements.txt            # Dependencias del proyecto (449 bytes)
├── test_ai.py                  # Script de verificación AI
├── test_system.py              # Script de verificación completa (6.2 KB)
├── README.md                   # Documentación completa (9.6 KB)
├── .env.example                # Ejemplo de variables de entorno (470 bytes)
│
└── (Generados por train_model.py):
    ├── modelo_depresion.h5         # Modelo LSTM entrenado
    ├── tokenizer.pickle            # Tokenizador de texto
    ├── model_config.pickle         # Configuración del modelo
    ├── confusion_matrix.png        # Visualización de métricas
    ├── roc_curve.png              # Curva ROC
    └── training_history.png       # Gráficas de entrenamiento
```

### 🎯 Arquitectura del Sistema

```
┌──────────────────────────────────────────────────────────────┐
│                   MINDSENTINEL SYSTEM                        │
└──────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ train_model  │    │    app.py    │    │   Streamlit  │
│    .py       │───>│              │───>│      UI      │
└──────────────┘    └──────────────┘    └──────────────┘
      │                     │
      │                     │
      ▼                     ▼
┌──────────────┐    ┌────────────────────────────────┐
│  Artefactos  │    │     Sistema Multi-Agente       │
│  Generados:  │    │  ┌────────────────────────┐    │
│              │    │  │ Agente 1: Clasificador │    │
│ • modelo.h5  │    │  │   (LSTM + TensorFlow)  │    │
│ • tokenizer  │    │  └────────────────────────┘    │
│ • config     │    │               │                │
└──────────────┘    │               ▼                │
                    │  ┌────────────────────────┐    │
                    │  │ Agente 2: Explicador   │    │
                    │  │   (Gemini Flash)       │    │
                    │  └────────────────────────┘    │
                    │               │                │
                    │               ▼                │
                    │  ┌────────────────────────┐    │
                    │  │ Agente 3: Supervisor   │    │
                    │  │   (Gemini Flash)       │    │
                    │  └────────────────────────┘    │
                    └────────────────────────────────┘
                                    │
                                    ▼
                        ┌──────────────────────────┐
                        │  Resultados al Usuario   │
                        │  • Predicción            │
                        │  • Explicación           │
                        │  • Recomendaciones       │
                        └──────────────────────────┘
```

---

## 🚀 Instalación

### Requisitos del Sistema

#### Hardware Mínimo
- CPU: Intel Core i5 o equivalente
- RAM: 8 GB
- Disco: 2 GB libres

#### Hardware Recomendado (Entrenamiento)
- GPU: NVIDIA con CUDA (opcional, acelera 10x)
- RAM: 16 GB
- Disco: 5 GB libres

#### Software
- Python 3.9 o superior
- pip (gestor de paquetes)
- Navegador web moderno

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/adref404/mindsentinel.git
cd mindsentinel
```

### Paso 2: Crear Entorno Virtual

```bash
# Python 3.9+ requerido
py -3.10 -m venv venv

# Activar entorno
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Paso 3: Crear archivo .env desde .env.example
```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Windows (CMD)
copy .env.example .env

# Linux / Mac
cp .env.example .env
```

### Paso 4: Instalar Dependencias

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

### Paso 5: Obtener API Key de Google Gemini

1. Ve a [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crea una API Key gratuita
3. Configúrala como variable de entorno:

```bash
# Linux/Mac
export GOOGLE_API_KEY='tu_api_key_aqui'

# Windows (PowerShell)
$env:GOOGLE_API_KEY='tu_api_key_aqui'

# O edita directamente app.py línea 44
```

### Paso 6: Verificación de Dependencias

```bash
# Test para asegurarte que el agente LLM funcionará
python test_ai.py

# Diagnóstico completo del sistema
python test_system.py
```

---

## 📊 Dataset

### Obtener el Dataset de Kaggle

**Dataset: Reddit Depression Dataset by Rishabh Kausish**
- Link: https://www.kaggle.com/datasets/rishabhkausish/reddit-depression-dataset
- Subreddits: teenagers, depression, suicidewatch, deepthoughts, happy, posts
- Labels: 1 (Depression/SuicideWatch) y 0 (Otros)

**Opción 1: Usar kagglehub (RECOMENDADO)**

```bash
# Instalar kagglehub
# pip install kagglehub # está en requirements.txt

# El script train_model.py descargará automáticamente el dataset con:
import kagglehub
path = kagglehub.dataset_download("rishabhkausish/reddit-depression-dataset")
```

**Opción 2: Descarga Manual**
1. Ve a [Reddit Depression Dataset](https://www.kaggle.com/datasets/rishabhkausish/reddit-depression-dataset)
2. Descarga el archivo CSV
3. Colócalo en el directorio del proyecto como `reddit_depression_dataset.csv`

**Opción 3: API de Kaggle**

```bash
# pip install kaggle
kaggle datasets download -d rishabhkausish/reddit-depression-dataset
unzip reddit-depression-dataset.zip
```

### Formato del Dataset

El archivo contiene estas columnas:
- `subreddit`: Subreddit donde se publicó el post
- `title`: Título del post
- `body`: Contenido del post
- `upvotes`: Número de upvotes
- `created_utc`: Timestamp de creación
- `number_of_comments`: Número de comentarios
- `label`: 0 (Normal) o 1 (Depresión)

---

## 🎓 Uso del Sistema

### Fase 1: Entrenamiento del Modelo

```bash
# Local (si tienes GPU)
python train_model.py

# Google Colab (recomendado)
# 1. Sube train_model.py a Colab
# 2. Sube depression_dataset.csv a Colab
# 3. Ejecuta el notebook
```

**Salida esperada:**
- `modelo_depresion.h5` - Modelo LSTM entrenado
- `tokenizer.pickle` - Tokenizador de texto
- `model_config.pickle` - Configuración del modelo
- `confusion_matrix.png` - Matriz de confusión
- `roc_curve.png` - Curva ROC
- `training_history.png` - Historial de métricas

**⏱️ Tiempo estimado:** 10-30 minutos (depende del tamaño del dataset y GPU)

### Fase 2: Ejecutar la Aplicación Multi-Agente

```bash
streamlit run app.py
```

Se abrirá automáticamente en tu navegador en `http://localhost:8501`

---

## 💻 Guía de Uso de la Interfaz

### 1. Entrada de Datos (En Inglés)

- **Título del Post**: Escribe un título simulado de Reddit
- **Subreddit**: Selecciona el contexto (r/depression, r/mentalhealth, etc.)
- **Cuerpo del Post**: Escribe el contenido completo (mínimo 20 caracteres)

**Ejemplo de entrada:**

```
Title: I can't go on like this
Subreddit: r/depression
Cuerpo: I don't feel anything anymore. 
Every day it gets harder to get out of bed. 
I don't see the point of trying anymore. 
I feel completely alone and empty. 
Everyone would be better off without me.
```

### 2. Análisis Multi-Agente

Al presionar **"Analizar con MindSentinel"**, el sistema ejecuta:

1. **Agente Clasificador**: Analiza el texto con el modelo LSTM
   - Genera probabilidad de depresión (0-100%)
   - Clasifica riesgo (BAJO / MEDIO / ALTO)

2. **Agente Explicador (XAI)**: Utiliza Gemini para explicar
   - Identifica palabras clave emocionales
   - Detecta patrones lingüísticos depresivos
   - Explica distorsiones cognitivas

3. **Agente Supervisor**: Toma decisión final
   - Evalúa coherencia de análisis previos
   - Genera recomendaciones específicas
   - Proporciona recursos de ayuda

---

## 🏗️ Arquitectura Técnica Detallada

### Modelo de Deep Learning

```python
Arquitectura LSTM Bidireccional:
- Embedding Layer (128 dim)
- SpatialDropout (20%)
- Bidirectional LSTM (64 units) × 2
- GlobalMaxPooling
- Dense (64) + BatchNorm + Dropout
- Dense (32) + BatchNorm + Dropout
- Dense (1, sigmoid)

Optimizador: Adam (lr=0.001)
Loss: Binary Crossentropy
Métricas: Accuracy, Precision, Recall, AUC-ROC
```

### Sistema Multi-Agente con CrewAI

**Agente 1: Clasificador**
- **Tecnología**: TensorFlow/Keras
- **Función**: Predicción numérica de riesgo
- **Salida**: Probabilidad + nivel de riesgo

**Agente 2: Explicador XAI**
- **Tecnología**: Google Gemini 2.5 Flash
- **Función**: Interpretabilidad del modelo
- **Salida**: Análisis lingüístico detallado

**Agente 3: Supervisor**
- **Tecnología**: Google Gemini 2.5 Flash
- **Función**: Decisión clínica final
- **Salida**: Recomendaciones + recursos

---

## 📈 Métricas de Rendimiento

| Métrica | Valor Esperado |
|---------|---------------|
| Accuracy | ~85-90% |
| Precision | ~82-88% |
| Recall | ~80-85% |
| AUC-ROC | ~0.88-0.92 |

*Valores varían según el dataset utilizado*

---

## 🎓 Casos de Uso Académico

### ✅ Ideal para:
- Tesis de maestría en IA/NLP
- Proyectos finales de ingeniería
- Investigación en salud mental digital
- Estudios de sistemas multi-agente
- Demostraciones de XAI (Explainable AI)

### 🔬 Áreas de Investigación:
- Natural Language Processing (NLP)
- Deep Learning para texto
- Sistemas Multi-Agente
- Inteligencia Artificial Explicable (XAI)
- Salud Mental Digital
- Detección temprana de riesgos psicológicos

---

## 🔧 Troubleshooting

### Error: "No se encontraron artefactos del modelo"

**Solución:** Ejecuta primero `train_model.py` para generar los archivos necesarios.

### Error: "GOOGLE_API_KEY no configurada"

**Solución:** 
```bash
export GOOGLE_API_KEY='tu_clave_aqui'
# O edita app.py línea 44
```

### Error: "ModuleNotFoundError: No module named 'crewai'"

**Solución:**
```bash
pip install -r requirements.txt
```

### La aplicación es muy lenta

**Solución:** 
- Gemini Flash es rápido, pero depende de tu conexión
- Considera usar un modelo local si necesitas más velocidad
- Verifica que el modelo .h5 esté cargado correctamente

---

## 🎨 Personalización

### Cambiar el Umbral de Riesgo

En `app.py`, línea 170:

```python
if probabilidad >= 0.7:  # Cambiar este valor
    nivel_riesgo = "ALTO"
elif probabilidad >= 0.4:  # Cambiar este valor
    nivel_riesgo = "MEDIO"
```

### Usar Otro Modelo LLM

Reemplaza en `app.py`:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    api_key="tu_api_key"
)
```

### Agregar Más Agentes

```python
def crear_agente_monitor():
    return Agent(
        role='Monitor de Seguimiento',
        goal='Rastrear evolución temporal',
        backstory='...',
        llm=llm
    )
```

---

## 📚 Referencias Académicas

1. **Coppersmith et al. (2015)** - "Quantifying Mental Health Signals in Twitter"
   
2. **Hochreiter & Schmidhuber (1997)** - "Long Short-Term Memory"
   
3. **Wooldridge & Jennings (1995)** - "Intelligent Agents"

4. **Holzinger et al. (2022)** - "Explainable AI in Healthcare"

---

## ⚠️ Consideraciones Éticas y Seguridad

### ❌ NO debe usarse para:
- Diagnóstico clínico real
- Sustitución de terapia profesional
- Decisiones médicas sin supervisión
- Vigilancia no consentida
- Reemplazo de atención médica profesional

### ✅ SÍ puede usarse para:
- Investigación académica controlada
- Desarrollo de herramientas de detección temprana
- Estudios de viabilidad técnica
- Educación en IA y salud mental
- Prototipado de sistemas de detección temprana

### 🔐 Seguridad y Privacidad

- **Datos**: Solo texto simulado, sin información personal real
- **API Keys**: Nunca compartir GOOGLE_API_KEY públicamente
- **Almacenamiento**: Los modelos se guardan localmente
- **GDPR/HIPAA**: No aplicable (proyecto académico sin datos reales)

---

## 📞 Recursos de Ayuda

### 🆘 Líneas de Crisis (en caso real)

- p🇪 Perú: **0800-10828** (Infosalud)
- 🇲🇽 México: **800 290 0024** (SAPTEL)
- 🇦🇷 Argentina: **135** (Centro de Asistencia al Suicida)
- 🇺🇸 USA: **988** (Suicide & Crisis Lifeline)
- 🌍 Internacional: [findahelpline.com](https://findahelpline.com)

### 🔗 Enlaces Útiles

- [Google AI Studio](https://makersuite.google.com/app/apikey) - API Key gratuita
- [Kaggle Dataset](https://www.kaggle.com/datasets/rishabhkausish/reddit-depression-dataset)
- [CrewAI Docs](https://docs.crewai.com)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Streamlit Documentation](https://docs.streamlit.io)

---

## 📈 Roadmap Futuro

### Posibles Mejoras:
- [ ] Integración con BERT o GPT para mejor precisión
- [ ] Análisis multimodal (texto + imágenes)
- [ ] Dashboard de monitoreo temporal
- [ ] API REST para integración con otras apps
- [ ] Soporte para más idiomas (actualmente: inglés)
- [ ] Detección de otras condiciones (ansiedad, PTSD)
- [ ] Sistema de feedback para mejorar el modelo

---

## 🤝 Contribuciones

Este es un proyecto académico abierto a mejoras:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -m 'Agregar mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

---

## 📄 Licencia

Uso académico e investigación. No usar en producción médica sin validación clínica.

---

## 🙏 Agradecimientos

- Comunidad de Kaggle por los datasets de salud mental
- Google por Gemini API gratuita
- CrewAI por el framework de agentes
- Comunidad de TensorFlow y Streamlit

---

## 🎉 ¡Comienza Ahora!

1. **Descarga** todos los archivos del proyecto
2. **Sigue** las instrucciones de instalación
3. **Entrena** el modelo con `train_model.py`
4. **Ejecuta** la aplicación con `streamlit run app.py`
5. **Explora** y mejora el sistema

---

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub**

---

**🧠 MindSentinel** - Arquitectura Multi-Agente para el Monitoreo de Salud Mental

*"Inteligencia Artificial al servicio del bienestar humano"*

---

**Última actualización:** Noviembre 2025  
**Versión:** 1.0  
**Autor:** fernando.celadita@unmsm.edu.pe