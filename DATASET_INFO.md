# 📊 Información del Dataset - Reddit Depression Dataset

## 📌 Información General

| Campo | Detalle |
|-------|---------|
| **Nombre** | Reddit Depression Dataset |
| **Autor** | Rishabh Kausish |
| **Fuente** | Kaggle |
| **Link** | https://www.kaggle.com/datasets/rishabhkausish/reddit-depression-dataset |
| **Tamaño** | ~7,000+ posts |
| **Formato** | CSV |

---

## 🗂️ Subreddits Incluidos

El dataset contiene posts de **6 subreddits** diferentes:

### Subreddits Etiquetados como **Depresión (Label = 1)**
1. **r/depression** - Comunidad de apoyo para personas con depresión
2. **r/SuicideWatch** - Comunidad de apoyo en crisis

### Subreddits Etiquetados como **Normal (Label = 0)**
3. **r/teenagers** - Conversaciones de adolescentes
4. **r/DeepThoughts** - Reflexiones filosóficas
5. **r/happy** - Posts sobre experiencias positivas
6. **r/posts** - Posts generales

---

## 📋 Estructura del Dataset

### Columnas del CSV

| Columna | Tipo | Descripción |
|---------|------|-------------|
| **Subreddit** | string | Nombre del subreddit donde se publicó |
| **Title** | string | Título del post de Reddit |
| **Body** | string | Contenido completo del post |
| **Upvotes** | int | Número de votos positivos recibidos |
| **Created UTC** | int | Timestamp de creación (epoch time) |
| **Number of Comments** | int | Cantidad de comentarios en el post |
| **Label** | int | Etiqueta: 0 (Normal) o 1 (Depresión) |

### Ejemplo de Registro

```csv
Subreddit,Title,Body,Upvotes,Created UTC,Number of Comments,Label
depression,"I feel empty","I don't know what to do anymore. Everything feels meaningless...",156,1609459200,23,1
happy,"Got my dream job!","After months of searching, I finally got hired at my dream company!",892,1609545600,45,0
```

---

## 📥 Métodos de Descarga

### Método 1: kagglehub (RECOMENDADO - Automático)

```python
import kagglehub

# Descarga automática del dataset
path = kagglehub.dataset_download("rishabhkausish/reddit-depression-dataset")
print("Path to dataset files:", path)

# El archivo CSV estará en: path/archivo.csv
```

**Ventajas:**
- ✅ Descarga automática
- ✅ No requiere configuración de credenciales
- ✅ Integrado en train_model.py

**Instalación:**
```bash
pip install kagglehub
```

---

### Método 2: Kaggle CLI

```bash
# 1. Instalar Kaggle CLI
pip install kaggle

# 2. Configurar credenciales
# Descarga tu kaggle.json desde: https://www.kaggle.com/settings/account
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Descargar dataset
kaggle datasets download -d rishabhkausish/reddit-depression-dataset

# 4. Descomprimir
unzip reddit-depression-dataset.zip
```

---

### Método 3: Descarga Manual

1. **Ve a la página del dataset:**
   https://www.kaggle.com/datasets/rishabhkausish/reddit-depression-dataset

2. **Inicia sesión en Kaggle** (crea cuenta gratuita si no tienes)

3. **Haz clic en "Download"** (botón azul en la esquina superior derecha)

4. **Descomprime el archivo** descargado

5. **Renombra el archivo** (si es necesario) a: `reddit_depression_dataset.csv`

6. **Coloca el archivo** en el directorio del proyecto `mindsentinel/`

---

## 📊 Estadísticas del Dataset

### Distribución de Clases

| Label | Descripción | Aproximado |
|-------|-------------|------------|
| 0 | Normal | ~50% |
| 1 | Depresión | ~50% |

El dataset está relativamente **balanceado**, lo cual es ideal para entrenamiento.

### Distribución por Subreddit

| Subreddit | Posts Aproximados | Label |
|-----------|------------------|-------|
| r/depression | ~2,500 | 1 |
| r/SuicideWatch | ~1,500 | 1 |
| r/teenagers | ~1,200 | 0 |
| r/DeepThoughts | ~800 | 0 |
| r/happy | ~700 | 0 |
| r/posts | ~300 | 0 |

---

## 🔍 Características del Texto

### Longitud de Posts

| Métrica | Valor Aproximado |
|---------|-----------------|
| Promedio | 150-300 palabras |
| Mínimo | 10 palabras |
| Máximo | 1000+ palabras |

### Características Lingüísticas

**Posts con Depresión (Label = 1):**
- ❌ Palabras negativas: "empty", "hopeless", "alone", "worthless"
- ❌ Primera persona singular: "I", "me", "myself"
- ❌ Tiempo presente: "feel", "am", "can't"
- ❌ Absolutos: "never", "always", "nothing", "everything"
- ❌ Temas: Soledad, desesperanza, ideación suicida

**Posts Normales (Label = 0):**
- ✅ Palabras positivas/neutras: "happy", "excited", "thinking"
- ✅ Diversidad de tiempos verbales
- ✅ Temas variados: Reflexiones, celebraciones, conversaciones casuales

---

## 🧹 Preprocesamiento Aplicado

En `train_model.py`, el texto pasa por:

1. **Combinación**: Title + Body
2. **Limpieza**:
   - Remover URLs (http, www)
   - Remover menciones (@username, u/username)
   - Remover links de subreddits (r/subreddit)
   - Mantener puntuación emocional (!, ?, ...)
   - Convertir a minúsculas
   - Remover números
   - Remover espacios múltiples

3. **Tokenización**: Convertir texto a secuencias numéricas
4. **Padding**: Normalizar longitud a 200 tokens

---

## ⚠️ Consideraciones Éticas

### Uso Apropiado
✅ **SÍ usar para:**
- Investigación académica en NLP
- Desarrollo de herramientas de detección temprana
- Estudios de viabilidad técnica
- Educación en IA y salud mental

### Uso NO Apropiado
❌ **NO usar para:**
- Diagnóstico clínico sin supervisión médica
- Vigilancia no consentida de usuarios
- Decisiones médicas sin validación profesional
- Discriminación o estigmatización

### Privacidad
- Los posts son públicos de Reddit
- No contienen información personal identificable
- Los nombres de usuario fueron anonimizados
- Timestamps fueron convertidos a epoch time

---

## 🔗 Referencias

### Dataset Original
- **Link**: https://www.kaggle.com/datasets/rishabhkausish/reddit-depression-dataset
- **Autor**: Rishabh Kausish
- **Licencia**: Verificar en la página de Kaggle

### Investigación Relacionada
- Coppersmith et al. (2015) - "Quantifying Mental Health Signals in Twitter"
- De Choudhury et al. (2013) - "Predicting Depression via Social Media"
- Yates et al. (2017) - "Depression and Self-Harm Risk Assessment in Online Forums"

---

## 📝 Código de Ejemplo para Carga

### Con kagglehub (Automático)

```python
import kagglehub
import pandas as pd
import os

# Descargar dataset
path = kagglehub.dataset_download("rishabhkausish/reddit-depression-dataset")

# Buscar archivo CSV
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
dataset_path = os.path.join(path, csv_files[0])

# Cargar con pandas
df = pd.read_csv(dataset_path)

print(f"Dataset cargado: {df.shape[0]} registros")
print(f"Columnas: {df.columns.tolist()}")
print(f"\nDistribución de clases:")
print(df['Label'].value_counts())
```

### Con archivo local

```python
import pandas as pd

# Cargar dataset local
df = pd.read_csv('reddit_depression_dataset.csv')

# Explorar datos
print(f"Total de posts: {len(df)}")
print(f"\nSubreddits únicos: {df['Subreddit'].nunique()}")
print(df['Subreddit'].value_counts())

# Ver ejemplo de post
print(f"\nEjemplo de post:")
print(f"Title: {df['Title'].iloc[0]}")
print(f"Body: {df['Body'].iloc[0][:200]}...")
print(f"Label: {df['Label'].iloc[0]} ({'Depresión' if df['Label'].iloc[0] == 1 else 'Normal'})")
```

---

## 🎯 Uso en MindSentinel

### Flujo de Trabajo

```
┌─────────────────────────────────────────────────────────┐
│  1. DESCARGA: kagglehub descarga automáticamente       │
│     el dataset de Kaggle                               │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  2. CARGA: pandas lee el CSV con 7 columnas            │
│     (Subreddit, Title, Body, Upvotes, etc.)            │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  3. PREPROCESAMIENTO: Combina Title + Body y limpia    │
│     el texto (remover URLs, menciones, etc.)           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  4. TOKENIZACIÓN: Convierte texto a secuencias         │
│     numéricas con vocabulario de 10,000 palabras       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  5. ENTRENAMIENTO: LSTM Bidireccional aprende          │
│     patrones lingüísticos de depresión                 │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  6. EVALUACIÓN: Métricas de precisión, recall, AUC    │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Troubleshooting

### Error: "No module named 'kagglehub'"
```bash
pip install kagglehub
```

### Error: "Dataset not found"
- Verifica conexión a internet
- Intenta descarga manual
- Revisa que el nombre del dataset sea correcto

### Error: "CSV has different columns"
- Verifica que descargaste el dataset correcto
- Comprueba la versión del dataset en Kaggle
- Revisa la estructura con: `df.columns.tolist()`

### Error: "Too many missing values"
```python
# Verificar valores nulos
print(df.isnull().sum())

# Limpiar valores nulos
df['Title'] = df['Title'].fillna('')
df['Body'] = df['Body'].fillna('')
df = df.dropna(subset=['Label'])
```

---

## 📊 Métricas de Calidad del Dataset

| Aspecto | Evaluación |
|---------|-----------|
| Tamaño | ⭐⭐⭐⭐ (7K+ posts) |
| Balance | ⭐⭐⭐⭐⭐ (~50/50) |
| Diversidad | ⭐⭐⭐⭐ (6 subreddits) |
| Limpieza | ⭐⭐⭐⭐ (pocos nulos) |
| Relevancia | ⭐⭐⭐⭐⭐ (casos reales) |

---

## 💡 Consejos para Mejor Rendimiento

1. **Combinar Title + Body**: Más contexto = mejor precisión
2. **No remover stopwords**: Palabras como "I", "no", "never" son importantes
3. **Mantener puntuación emocional**: !!!, ???, ... indican intensidad
4. **Balancear clases**: Usar class_weight en entrenamiento
5. **Validación estratificada**: Mantener proporción 50/50 en splits

---

## 🎓 Citación

Si usas este dataset en investigación académica:

```
Kausish, R. (2023). Reddit Depression Dataset. 
Kaggle. https://www.kaggle.com/datasets/rishabhkausish/reddit-depression-dataset
```

---

**📊 Dataset listo para ser usado en MindSentinel**

Para comenzar: `python train_model.py` (descarga automática incluida)
