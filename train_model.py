"""
MindSentinel - Script de Entrenamiento del Modelo de Deep Learning
=====================================================================
Arquitectura: LSTM Bidireccional con Embeddings pre-entrenados
Dataset: Reddit Depression Dataset (Kaggle)
Autor: Arquitecto de Software Senior
Compatible con: Google Colab
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
# 1. CARGA Y EXPLORACIÓN DEL DATASET
# ============================================================================
print("\n[1/7] Cargando dataset...")

# En Google Colab, primero debes subir el archivo o conectar con Kaggle
# Opción 1: Subida manual
# from google.colab import files
# uploaded = files.upload()

# Opción 2: Descarga directa desde Kaggle (recomendado)
# !pip install -q kaggle
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d infamouscoder/mental-health-social-media
# !unzip mental-health-social-media.zip

# Carga del dataset
try:
    df = pd.read_csv('depression_dataset.csv')
    print(f"✓ Dataset cargado: {df.shape[0]} registros, {df.shape[1]} columnas")
except FileNotFoundError:
    print("❌ Error: No se encontró 'depression_dataset.csv'")
    print("📌 Asegúrate de tener el archivo en el directorio actual")
    exit(1)

# Exploración básica
print(f"\n📊 Distribución de clases:")
print(df['Label'].value_counts())
print(f"\n📈 Proporción:")
print(df['Label'].value_counts(normalize=True))

# Verificar valores nulos
print(f"\n🔍 Valores nulos: {df.isnull().sum().sum()}")
df = df.dropna(subset=['Body', 'Label'])

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
    
    # Mantener puntuación emocional (!, ?, ...)
    # Remover otros caracteres especiales pero mantener apóstrofes
    text = re.sub(r'[^\w\s!?.\']', ' ', text)
    
    # Remover números (opcional, depende del contexto)
    text = re.sub(r'\d+', '', text)
    
    # Remover espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Aplicar limpieza
df['clean_text'] = df['Body'].apply(clean_text)

# Filtrar textos muy cortos (menos de 10 caracteres)
df = df[df['clean_text'].str.len() > 10]

print(f"✓ Preprocesamiento completado. Registros válidos: {len(df)}")
print(f"📝 Ejemplo de texto limpio:\n{df['clean_text'].iloc[0][:200]}...")

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
y = df['Label'].values

print(f"✓ Vocabulario creado: {len(tokenizer.word_index)} palabras únicas")
print(f"✓ Secuencias generadas: {X.shape}")
print(f"✓ Palabras más frecuentes: {list(tokenizer.word_index.items())[:10]}")

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

print(f"✓ Train: {X_train.shape[0]} muestras ({y_train.mean()*100:.1f}% positivos)")
print(f"✓ Validation: {X_val.shape[0]} muestras ({y_val.mean()*100:.1f}% positivos)")
print(f"✓ Test: {X_test.shape[0]} muestras ({y_test.mean()*100:.1f}% positivos)")

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

# Mostrar arquitectura
print("\n📐 Arquitectura del modelo:")
model.summary()

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

print(f"⚖️ Pesos de clase: Negativo={neg_weight:.2f}, Positivo={pos_weight:.2f}")

# Entrenar modelo
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
print(f"\n🎯 Accuracy: {test_acc:.4f}")
print(f"🎯 Precision: {test_precision:.4f}")
print(f"🎯 Recall: {test_recall:.4f}")
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
print("✓ Matriz de confusión guardada: confusion_matrix.png")

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
    'test_recall': float(test_recall)
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
   1. modelo_depresion.h5 ............ Modelo LSTM entrenado
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

📌 Próximos pasos:
   1. Descarga estos archivos desde Colab
   2. Úsalos en el script app.py con Streamlit
   3. Los agentes de CrewAI utilizarán estos artefactos para predicción
""")

print("=" * 80)
print("🧠 MindSentinel está listo para monitorear la salud mental")
print("=" * 80)
