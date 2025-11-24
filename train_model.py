"""
MindSentinel - Script de Entrenamiento del Modelo de Deep Learning
=====================================================================
Arquitectura: LSTM Bidireccional con Embeddings pre-entrenados
Dataset: Reddit Depression Dataset (Kaggle - rishabhkausish)
Subreddits: teenagers, depression, suicidewatch, deepthoughts, happy, posts
Autor: Arquitecto de Software Senior
Compatible con: Google Colab / Local
"""

import numpy as np
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Bidirectional, Dense, Dropout, 
    SpatialDropout1D, GlobalMaxPooling1D, BatchNormalization,
    Input, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Configuración de seeds para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("🧠 MINDSENTINEL - ENTRENAMIENTO DE MODELO DE DETECCIÓN DE DEPRESIÓN")
print("=" * 80)

# ============================================================================
# 1. DESCARGA Y CARGA DEL DATASET
# ============================================================================
print("\n[1/7] Descargando y cargando dataset desde Kaggle...")

# Opción 1: Usar kagglehub (RECOMENDADO)
try:
    import kagglehub
    import os
    print("📥 Descargando dataset con kagglehub...")
    path = kagglehub.dataset_download("rishabhkausish/reddit-depression-dataset")
    print(f"✓ Dataset descargado en: {path}")
    
    # Buscar el archivo CSV en el path descargado
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    if csv_files:
        dataset_path = os.path.join(path, csv_files[0])
        print(f"✓ Archivo encontrado: {csv_files[0]}")
    else:
        print("❌ No se encontró archivo CSV en el directorio descargado")
        dataset_path = 'reddit_depression_dataset.csv'  # Fallback
        
except ImportError:
    print("⚠️  kagglehub no instalado. Instala con: pip install kagglehub")
    print("📌 Buscando archivo local...")
    dataset_path = 'reddit_depression_dataset.csv'
except Exception as e:
    print(f"⚠️  Error al descargar: {e}")
    print("📌 Buscando archivo local...")
    dataset_path = 'reddit_depression_dataset.csv'

# Opción 2: Si ya tienes el archivo local
# dataset_path = 'reddit_depression_dataset.csv'

# Cargar dataset
try:
    df = pd.read_csv(dataset_path, low_memory=False)
    print(f"\n✓ Dataset cargado exitosamente")
    print(f"\n✓ Dataset cargado. Dimensiones originales: {df.shape}")

    # TRUCO: Convertir todo a minúsculas para evitar errores de 'Subreddit' vs 'subreddit'
    df.columns = df.columns.str.lower()
    
    # Renombrar columnas para estandarizar
    # Mapeamos 'body' -> 'text' (input) y 'label' -> 'label' (target)
    rename_map = {
        'body': 'text',
        'title': 'title',
        'subreddit': 'source', 
        'label': 'label'
    }
    df = df.rename(columns=rename_map)
    
    # Verificar que existen las columnas críticas
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"Faltan columnas críticas. Disponibles: {df.columns.tolist()}")
    
except FileNotFoundError:
    print(f"\n❌ Error: No se encontró '{dataset_path}'")
    print("\n📌 Opciones de solución:")
    print("   1. Instala kagglehub: pip install kagglehub")
    print("   2. Descarga manualmente desde: https://www.kaggle.com/datasets/rishabhkausish/reddit-depression-dataset")
    print("   3. Coloca el archivo CSV en el directorio actual")
    exit(1)

# --- 3. REDUCCIÓN DEL DATASET (SAMPLING) ---
# IMPORTANTE: 2.4 Millones de filas es demasiado para entrenar rápido.
# Vamos a tomar una muestra balanceada de 20,000 registros (10k depresión, 10k normal).

print("\n✂️ Aplicando Sampling (Reducción de datos para eficiencia)...")
SAMPLE_SIZE = 10000  # 10k por clase = 20k total. (Sube esto si tienes GPU potente)

df_dep = df[df['label'] == 1]
df_norm = df[df['label'] == 0]

# Tomar muestra aleatoria
if len(df_dep) > SAMPLE_SIZE:
    df_dep = df_dep.sample(n=SAMPLE_SIZE, random_state=42)
if len(df_norm) > SAMPLE_SIZE:
    df_norm = df_norm.sample(n=SAMPLE_SIZE, random_state=42)

# Combinar y mezclar (shuffle)
df = pd.concat([df_dep, df_norm])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Limpieza básica de nulos en el texto final
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str) # Asegurar que sea string

print(f"✓ Dataset reducido y listo: {len(df)} registros")
print("-" * 30)
print(df['label'].value_counts())
print("-" * 30)

# --- 4. CONTINUAR FLUJO ---
# Variables finales para el resto del script
print(f"\n📋 Columnas finales usadas: {df.columns.tolist()}")
print(f"📊 Ejemplo de texto:\n {df['text'].iloc[0][:100]}...")


# ============================================================================
# 2. PREPROCESAMIENTO DE TEXTO NLP
# ============================================================================
print("\n[2/7] Preprocesando texto...")

def clean_text(text):
    """
    Limpieza avanzada de texto para análisis de salud mental
    
    Mantiene: palabras emocionales, negaciones, puntuación significativa
    Elimina: URLs, menciones, caracteres especiales innecesarios
    """
    if not isinstance(text, str):
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Remover URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remover menciones de usuarios (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remover subreddit links (r/subreddit)
    text = re.sub(r'r/\w+', '', text)
    
    # Remover u/username
    text = re.sub(r'u/\w+', '', text)
    
    # Mantener puntuación emocional (!, ?, ...)
    # Remover otros caracteres especiales pero mantener apóstrofes
    text = re.sub(r'[^\w\s!?.\']', ' ', text)
    
    # Remover números (opcional, depende del contexto)
    text = re.sub(r'\d+', '', text)
    
    # Remover espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Combinar title y text (body) para análisis completo
# CORRECCIÓN: usar 'text' en lugar de 'Body' (ya renombrado a minúsculas)
print("📝 Combinando title + text...")

# Manejar valores nulos antes de combinar
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')

df['combined_text'] = df['title'].astype(str) + ". " + df['text'].astype(str)

# Aplicar limpieza
print("🧹 Limpiando texto...")
df['clean_text'] = df['combined_text'].apply(clean_text)

# Filtrar textos muy cortos (menos de 10 caracteres)
df = df[df['clean_text'].str.len() > 10]

print(f"\n✓ Preprocesamiento completado. Registros válidos: {len(df)}")
print(f"\n📊 Estadísticas de longitud de texto:")
print(f"   Media: {df['clean_text'].str.len().mean():.0f} caracteres")
print(f"   Mediana: {df['clean_text'].str.len().median():.0f} caracteres")
print(f"   Máximo: {df['clean_text'].str.len().max():.0f} caracteres")

print(f"\n📝 Ejemplo de texto limpio:")
print(f"Original (title): {df['title'].iloc[0][:100]}...")
print(f"Original (text): {df['text'].iloc[0][:100]}...")
print(f"Limpio: {df['clean_text'].iloc[0][:200]}...")

# ============================================================================
# 3. TOKENIZACIÓN Y SECUENCIACIÓN
# ============================================================================
print("\n[3/7] Tokenizando y creando secuencias...")

# Hiperparámetros
MAX_WORDS = 10000  # Vocabulario máximo
MAX_LEN = 200      # Longitud máxima de secuencia
EMBEDDING_DIM = 128

# Crear tokenizador
tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>",  # Token para palabras fuera de vocabulario
    lower=True
)

# Entrenar tokenizador con el corpus
tokenizer.fit_on_texts(df['clean_text'])

# Convertir textos a secuencias de números
sequences = tokenizer.texts_to_sequences(df['clean_text'])

# Padding: hacer todas las secuencias de la misma longitud
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

# Labels
y = df['label'].values

print(f"✓ Vocabulario creado: {len(tokenizer.word_index)} palabras únicas")
print(f"✓ Secuencias generadas: {X.shape}")
print(f"✓ Vocabulario usado: {min(MAX_WORDS, len(tokenizer.word_index))} palabras")
print(f"\n🔤 Palabras más frecuentes:")
word_freq = sorted(tokenizer.word_index.items(), key=lambda x: x[1])[:15]
print(f"   {word_freq}")

# ============================================================================
# 4. DIVISIÓN DEL DATASET
# ============================================================================
print("\n[4/7] Dividiendo dataset en train/validation/test...")

# División estratificada para mantener proporción de clases
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)  # 0.176 de 0.85 ≈ 0.15 del total

print(f"✓ Train set: {X_train.shape[0]} muestras ({y_train.mean()*100:.1f}% depresión)")
print(f"✓ Validation set: {X_val.shape[0]} muestras ({y_val.mean()*100:.1f}% depresión)")
print(f"✓ Test set: {X_test.shape[0]} muestras ({y_test.mean()*100:.1f}% depresión)")

# ============================================================================
# 5. CONSTRUCCIÓN DEL MODELO DE DEEP LEARNING
# ============================================================================
print("\n[5/7] Construyendo arquitectura LSTM Bidireccional...")

def build_depression_detection_model():
    """
    Arquitectura optimizada para detección de depresión en texto
    
    Componentes:
    - Embedding Layer: Representación vectorial de palabras
    - SpatialDropout1D: Regularización para embeddings
    - Bidirectional LSTM: Captura contexto en ambas direcciones
    - GlobalMaxPooling: Extrae características más importantes
    - Dense Layers: Clasificación final
    - Batch Normalization: Estabiliza entrenamiento
    """
    model = Sequential([
        # Capa de Embedding
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_LEN,
            name='embedding_layer'
        ),
        
        # Dropout espacial (más efectivo para embeddings)
        SpatialDropout1D(0.2),
        
        # Primera LSTM Bidireccional
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        
        # Segunda LSTM Bidireccional
        Bidirectional(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
        
        # Global Max Pooling para capturar características clave
        GlobalMaxPooling1D(),
        
        # Capas densas con regularización
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Capa de salida (clasificación binaria)
        Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    return model

# Crear modelo
model = build_depression_detection_model()

# Compilar con métricas relevantes
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)
# CORRECCIÓN: Construir el modelo antes de mostrar el resumen
model.build(input_shape=(None, MAX_LEN))

# Mostrar arquitectura
print("\n📐 Arquitectura del modelo:")
model.summary()

# Contar parámetros
total_params = model.count_params()
print(f"\n📊 Total de parámetros entrenables: {total_params:,}")

# ============================================================================
# 6. ENTRENAMIENTO DEL MODELO
# ============================================================================
print("\n[6/7] Entrenando modelo...")

# Callbacks para optimizar entrenamiento
callbacks = [
    # Early Stopping: detiene si no mejora en 5 épocas
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce Learning Rate: disminuye LR si se estanca
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    ),
    
    # Model Checkpoint: guarda mejor modelo
    ModelCheckpoint(
        'best_model_checkpoint.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# Calcular pesos de clase para balancear dataset desbalanceado
neg_weight = (1 / np.sum(y_train == 0)) * (len(y_train) / 2.0)
pos_weight = (1 / np.sum(y_train == 1)) * (len(y_train) / 2.0)
class_weight = {0: neg_weight, 1: pos_weight}

print(f"\n⚖️  Pesos de clase (para balanceo):")
print(f"   Clase 0 (Normal): {neg_weight:.2f}")
print(f"   Clase 1 (Depresión): {pos_weight:.2f}")

# Entrenar modelo
print("\n🚀 Iniciando entrenamiento...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Entrenamiento completado!")

# ============================================================================
# 7. EVALUACIÓN Y GUARDADO DE ARTEFACTOS
# ============================================================================
print("\n[7/7] Evaluando modelo y guardando artefactos...")

# Predicciones en conjunto de test
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Métricas de evaluación
print("\n" + "=" * 80)
print("📊 MÉTRICAS DE EVALUACIÓN EN TEST SET")
print("=" * 80)

test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"🎯 Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"🎯 Recall: {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"🎯 AUC-ROC: {test_auc:.4f}")

# Reporte de clasificación detallado
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Depresión']))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Depresión'],
            yticklabels=['Normal', 'Depresión'])
plt.title('Matriz de Confusión - MindSentinel', fontsize=14, fontweight='bold')
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Matriz de confusión guardada: confusion_matrix.png")

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - MindSentinel', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Curva ROC guardada: roc_curve.png")

# Gráficas de entrenamiento
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0, 0].set_title('Accuracy durante entrenamiento')
axes[0, 0].set_xlabel('Época')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train Loss')
axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
axes[0, 1].set_title('Loss durante entrenamiento')
axes[0, 1].set_xlabel('Época')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train Precision')
axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
axes[1, 0].set_title('Precision durante entrenamiento')
axes[1, 0].set_xlabel('Época')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train Recall')
axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
axes[1, 1].set_title('Recall durante entrenamiento')
axes[1, 1].set_xlabel('Época')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✓ Historial de entrenamiento guardado: training_history.png")

# ============================================================================
# GUARDAR ARTEFACTOS PARA PRODUCCIÓN
# ============================================================================
print("\n" + "=" * 80)
print("💾 GUARDANDO ARTEFACTOS PARA PRODUCCIÓN")
print("=" * 80)

# 1. Guardar modelo entrenado
model.save('modelo_depresion.h5')
print("✓ Modelo guardado: modelo_depresion.h5")

# 2. Guardar tokenizador
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("✓ Tokenizador guardado: tokenizer.pickle")

# 3. Guardar configuración del modelo
config = {
    'max_words': MAX_WORDS,
    'max_len': MAX_LEN,
    'embedding_dim': EMBEDDING_DIM,
    'vocab_size': len(tokenizer.word_index),
    'test_accuracy': float(test_acc),
    'test_auc': float(test_auc),
    'test_precision': float(test_precision),
    'test_recall': float(test_recall),
    'dataset_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'subreddits': df['source'].unique().tolist()
    }
}

with open('model_config.pickle', 'wb') as handle:
    pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("✓ Configuración guardada: model_config.pickle")

# ============================================================================
# RESUMEN FINAL
# ============================================================================
print("\n" + "=" * 80)
print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("=" * 80)
print(f"""
📦 Artefactos generados:
   1. modelo_depresion.h5 ............ Modelo LSTM entrenado ({model.count_params():,} parámetros)
   2. tokenizer.pickle ............... Tokenizador de texto
   3. model_config.pickle ............ Configuración del modelo
   4. confusion_matrix.png ........... Visualización de métricas
   5. roc_curve.png .................. Curva ROC
   6. training_history.png ........... Historial de entrenamiento

📊 Métricas finales:
   • Accuracy: {test_acc:.2%}
   • AUC-ROC: {test_auc:.2%}
   • Precision: {test_precision:.2%}
   • Recall: {test_recall:.2%}

📈 Información del dataset:
   • Total de muestras: {len(df):,}
   • Subreddits analizados: {len(df['source'].unique())}
   • {', '.join(df['source'].unique())}

📌 Próximos pasos:
   1. Descarga estos archivos desde Colab (si aplica)
   2. Úsalos en el script app.py con Streamlit
   3. Los agentes de CrewAI utilizarán estos artefactos para predicción
""")

print("=" * 80)
print("🧠 MindSentinel está listo para monitorear la salud mental")
print("=" * 80)
