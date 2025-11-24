# 🔗 Instrucciones de Conexión entre Scripts

## 📦 Archivos del Proyecto

Tu proyecto **MindSentinel** consta de los siguientes archivos:

```
mindsentinel/
│
├── train_model.py              # Script de entrenamiento (Parte 1)
├── app.py                      # Aplicación multi-agente (Parte 2)
├── requirements.txt            # Dependencias del proyecto
├── test_system.py             # Script de verificación
├── README.md                   # Documentación completa
├── .env.example               # Ejemplo de variables de entorno
│
└── (Generados por train_model.py):
    ├── modelo_depresion.h5         # Modelo LSTM entrenado
    ├── tokenizer.pickle            # Tokenizador de texto
    ├── model_config.pickle         # Configuración del modelo
    ├── confusion_matrix.png        # Visualización de métricas
    ├── roc_curve.png              # Curva ROC
    └── training_history.png       # Gráficas de entrenamiento
```

---

## 🔄 Flujo de Trabajo Completo

### FASE 1: Preparación del Entorno

#### Paso 1.1: Crear Directorio del Proyecto

```bash
mkdir mindsentinel
cd mindsentinel
```

#### Paso 1.2: Guardar los Scripts

Copia estos 6 archivos en el directorio:
- `train_model.py`
- `app.py`
- `requirements.txt`
- `test_system.py`
- `README.md`
- `.env.example`

#### Paso 1.3: Crear Entorno Virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Linux/Mac)
source venv/bin/activate
```

#### Paso 1.4: Instalar Dependencias

```bash
pip install -r requirements.txt
```

#### Paso 1.5: Verificar Instalación

```bash
python test_system.py
```

---

### FASE 2: Obtener y Preparar Dataset

#### Opción A: Kaggle API (Recomendado)

```bash
# Instalar CLI de Kaggle
pip install kaggle

# Configurar credenciales (coloca tu kaggle.json en ~/.kaggle/)
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Descargar dataset
kaggle datasets download -d infamouscoder/mental-health-social-media
unzip mental-health-social-media.zip
```

#### Opción B: Descarga Manual

1. Ve a: https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media
2. Descarga `depression_dataset.csv`
3. Colócalo en el directorio `mindsentinel/`

#### Verificar Dataset

```bash
# Debe existir este archivo:
ls -lh depression_dataset.csv

# Verificar primeras líneas
head -5 depression_dataset.csv
```

---

### FASE 3: Entrenar el Modelo (Script 1)

#### Paso 3.1: Ejecutar Entrenamiento

```bash
# Local (si tienes GPU)
python train_model.py

# Google Colab (recomendado)
# 1. Sube train_model.py a Colab
# 2. Sube depression_dataset.csv a Colab
# 3. Ejecuta el notebook
```

#### Paso 3.2: Esperar Resultados

**Tiempo estimado:** 10-30 minutos

**Salida esperada:**
```
✓ Dataset cargado: 7731 registros, 2 columnas
✓ Preprocesamiento completado. Registros válidos: 7500
✓ Vocabulario creado: 25000 palabras únicas
✓ Train: 5422 muestras (52.3% positivos)
✓ Validation: 1153 muestras (52.5% positivos)
✓ Test: 1155 muestras (52.1% positivos)
...
🎯 Accuracy: 0.8756
🎯 Precision: 0.8432
🎯 Recall: 0.8234
🎯 AUC-ROC: 0.9123
```

#### Paso 3.3: Verificar Artefactos Generados

```bash
ls -lh modelo_depresion.h5
ls -lh tokenizer.pickle
ls -lh model_config.pickle
```

**Si usaste Google Colab, descarga estos 3 archivos:**

```python
# En Colab, ejecuta:
from google.colab import files
files.download('modelo_depresion.h5')
files.download('tokenizer.pickle')
files.download('model_config.pickle')
```

#### Paso 3.4: Copiar Artefactos al Directorio Local

Si entrenaste en Colab, copia los archivos descargados a tu directorio `mindsentinel/`:

```bash
# Deben estar en el mismo directorio que app.py
mindsentinel/
├── modelo_depresion.h5      ← Debe existir
├── tokenizer.pickle          ← Debe existir
├── model_config.pickle       ← Debe existir
└── app.py
```

---

### FASE 4: Configurar API de Google Gemini

#### Paso 4.1: Obtener API Key

1. Ve a: https://makersuite.google.com/app/apikey
2. Haz clic en "Create API Key"
3. Copia tu API Key (ejemplo: `AIzaSyD...`)

#### Paso 4.2: Configurar la API Key

**Opción A: Variable de Entorno (Recomendado)**

```bash
# Linux/Mac
export GOOGLE_API_KEY='AIzaSyD...'

# Windows PowerShell
$env:GOOGLE_API_KEY='AIzaSyD...'

# Windows CMD
set GOOGLE_API_KEY=AIzaSyD...
```

**Opción B: Editar app.py Directamente**

Abre `app.py` y edita la línea 44:

```python
# Línea 44 de app.py
GOOGLE_API_KEY = "AIzaSyD..."  # 👈 Pega tu API Key aquí
```

**Opción C: Archivo .env**

```bash
# Crear archivo .env
cp .env.example .env

# Editar .env
nano .env

# Agregar:
GOOGLE_API_KEY=AIzaSyD...
```

---

### FASE 5: Ejecutar la Aplicación (Script 2)

#### Paso 5.1: Verificar Sistema Completo

```bash
python test_system.py
```

**Salida esperada:**
```
✓ Python 3.10.12
✓ TensorFlow 2.15.0
✓ Streamlit 1.29.0
✓ CrewAI instalado
✓ LangChain Google GenAI instalado
✓ modelo_depresion.h5 (45.23 MB)
✓ tokenizer.pickle (2.45 MB)
✓ model_config.pickle (0.01 MB)
✓ GOOGLE_API_KEY configurada (39 caracteres)

🎉 ¡TODOS LOS COMPONENTES ESTÁN LISTOS!
```

#### Paso 5.2: Lanzar Streamlit

```bash
streamlit run app.py
```

Se abrirá automáticamente en: `http://localhost:8501`

---

## 🔗 Conexión entre Scripts: Puntos Clave

### 1. **Artefactos Compartidos**

El **Script 1** (train_model.py) genera:
- `modelo_depresion.h5` → Usado en app.py línea 134
- `tokenizer.pickle` → Usado en app.py línea 137
- `model_config.pickle` → Usado en app.py línea 140

### 2. **Preprocesamiento Idéntico**

**CRÍTICO:** El preprocesamiento debe ser idéntico en ambos scripts.

**train_model.py (líneas 68-90):**
```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    # ... resto del preprocesamiento
    return text
```

**app.py (líneas 146-160):**
```python
def clean_text(text):
    # DEBE SER IDÉNTICO A train_model.py
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # ...
    return text
```

### 3. **Configuración del Modelo**

**train_model.py** guarda configuración:
```python
config = {
    'max_words': 10000,
    'max_len': 200,
    'embedding_dim': 128,
    # ...
}
```

**app.py** la carga:
```python
config = pickle.load('model_config.pickle')
# Usa config['max_len'] en línea 168
```

### 4. **Flujo de Datos**

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAIN_MODEL.PY                           │
│  (Ejecutar UNA VEZ para entrenar)                          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  Genera 3 archivos:            │
          │  1. modelo_depresion.h5        │
          │  2. tokenizer.pickle           │
          │  3. model_config.pickle        │
          └────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       APP.PY                                │
│  (Ejecutar CADA VEZ que quieras usar la app)               │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  Carga los 3 archivos          │
          │  (líneas 134-140)              │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  Usuario ingresa texto         │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  Agente 1: Clasificador        │
          │  (usa modelo + tokenizer)      │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  Agente 2: Explicador XAI      │
          │  (usa Gemini)                  │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  Agente 3: Supervisor          │
          │  (usa Gemini)                  │
          └────────────────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────┐
          │  Muestra resultados en UI      │
          └────────────────────────────────┘
```

---

## 🧪 Prueba de Funcionamiento

### Ejemplo de Texto de Prueba

Una vez que la aplicación esté corriendo, prueba con:

**Entrada (Riesgo Alto):**
```
Título: No puedo seguir así
Subreddit: r/depression
Cuerpo: Ya no siento nada. Cada día es más difícil levantarme 
de la cama. No veo el punto de seguir intentando. Todos estarían 
mejor sin mí. No tengo energía ni para las cosas que antes amaba. 
Me siento completamente solo y vacío.
```

**Salida Esperada:**
- Probabilidad: ~85-95%
- Nivel: ALTO RIESGO 🔴
- Explicación XAI: Identificará palabras como "vacío", "solo", "no puedo", etc.
- Supervisor: Generará alerta clínica con recursos de ayuda

---

## ❓ Troubleshooting de Conexión

### Error: "No se encontraron artefactos del modelo"

**Causa:** app.py no encuentra los archivos .h5, .pickle

**Solución:**
```bash
# Verifica que los archivos existen en el mismo directorio
ls -la modelo_depresion.h5
ls -la tokenizer.pickle
ls -la model_config.pickle

# Si no existen, ejecuta train_model.py primero
python train_model.py
```

### Error: "Failed to load model"

**Causa:** Versión de TensorFlow diferente entre entrenamiento y producción

**Solución:**
```bash
# Usa la misma versión de TensorFlow
pip install tensorflow==2.15.0
```

### Error: "Text cleaning produces different results"

**Causa:** Las funciones clean_text() son diferentes en ambos scripts

**Solución:**
```python
# Asegúrate de que clean_text() sea IDÉNTICA
# en train_model.py y app.py
```

### Error: "GOOGLE_API_KEY not configured"

**Solución:**
```bash
export GOOGLE_API_KEY='tu_clave_aqui'
```

---

## 🎯 Checklist Final

Antes de ejecutar app.py, verifica:

- [ ] ✅ Python 3.9+ instalado
- [ ] ✅ Entorno virtual activado
- [ ] ✅ Todas las dependencias instaladas (`pip install -r requirements.txt`)
- [ ] ✅ Dataset `depression_dataset.csv` descargado
- [ ] ✅ `train_model.py` ejecutado exitosamente
- [ ] ✅ Archivos generados:
  - [ ] modelo_depresion.h5
  - [ ] tokenizer.pickle
  - [ ] model_config.pickle
- [ ] ✅ GOOGLE_API_KEY configurada
- [ ] ✅ `test_system.py` ejecutado sin errores
- [ ] ✅ Streamlit funciona correctamente

---

## 🚀 Comandos Rápidos (Resumen)

```bash
# 1. Preparar entorno
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# 2. Descargar dataset
kaggle datasets download -d infamouscoder/mental-health-social-media
unzip mental-health-social-media.zip

# 3. Entrenar modelo (una sola vez)
python train_model.py

# 4. Configurar API
export GOOGLE_API_KEY='tu_api_key'

# 5. Verificar sistema
python test_system.py

# 6. Ejecutar aplicación
streamlit run app.py
```

---

## 📊 Diagrama de Arquitectura Completa

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
┌──────────────┐    ┌──────────────────────────────────┐
│  Artefactos  │    │     Sistema Multi-Agente        │
│  Generados:  │    │  ┌────────────────────────┐    │
│              │    │  │ Agente 1: Clasificador │    │
│ • modelo.h5  │    │  │   (LSTM + TensorFlow)  │    │
│ • tokenizer  │    │  └────────────────────────┘    │
│ • config     │    │            │                    │
└──────────────┘    │            ▼                    │
                    │  ┌────────────────────────┐    │
                    │  │ Agente 2: Explicador   │    │
                    │  │   (Gemini Flash)       │    │
                    │  └────────────────────────┘    │
                    │            │                    │
                    │            ▼                    │
                    │  ┌────────────────────────┐    │
                    │  │ Agente 3: Supervisor   │    │
                    │  │   (Gemini Flash)       │    │
                    │  └────────────────────────┘    │
                    └──────────────────────────────────┘
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

## 💡 Consejos Finales

1. **Entrena el modelo solo UNA vez** (a menos que cambies el dataset)
2. **Guarda los artefactos** (.h5, .pickle) en un lugar seguro
3. **No compartas tu GOOGLE_API_KEY** públicamente
4. **Usa GPU para entrenar** (Google Colab gratuito tiene GPU)
5. **Prueba con textos variados** para validar el sistema
6. **Revisa los logs de CrewAI** para debugging
7. **Monitorea el uso de la API de Gemini** (tiene límites gratuitos)

---

## 📞 Soporte

Si encuentras errores:
1. Ejecuta `python test_system.py`
2. Revisa los logs de errores
3. Verifica que las versiones de librerías coincidan
4. Consulta la documentación oficial de CrewAI y Gemini

---

**¡Listo! Tu sistema MindSentinel debería estar funcionando perfectamente.**

🧠 **MindSentinel** - Arquitectura Multi-Agente para Monitoreo de Salud Mental
