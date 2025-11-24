"""
MindSentinel - Script de Prueba Rápida del Sistema
===================================================
Verifica que todos los componentes estén correctamente instalados
"""

import sys
import os

print("=" * 80)
print("🧪 MINDSENTINEL - TEST DE COMPONENTES")
print("=" * 80)

# ============================================================================
# 1. Verificar Python
# ============================================================================
print("\n[1/8] Verificando versión de Python...")
python_version = sys.version_info
if python_version.major == 3 and python_version.minor >= 9:
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
else:
    print(f"❌ Python {python_version.major}.{python_version.minor} detectado. Se requiere Python 3.9+")
    sys.exit(1)

# ============================================================================
# 2. Verificar TensorFlow
# ============================================================================
print("\n[2/8] Verificando TensorFlow...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except ImportError:
    print("❌ TensorFlow no instalado. Ejecuta: pip install tensorflow")
    sys.exit(1)

# ============================================================================
# 3. Verificar Streamlit
# ============================================================================
print("\n[3/8] Verificando Streamlit...")
try:
    import streamlit as st
    print(f"✓ Streamlit {st.__version__}")
except ImportError:
    print("❌ Streamlit no instalado. Ejecuta: pip install streamlit")
    sys.exit(1)

# ============================================================================
# 4. Verificar CrewAI
# ============================================================================
print("\n[4/8] Verificando CrewAI...")
try:
    import crewai
    print(f"✓ CrewAI instalado")
except ImportError:
    print("❌ CrewAI no instalado. Ejecuta: pip install crewai")
    sys.exit(1)

# ============================================================================
# 5. Verificar LangChain Google
# ============================================================================
print("\n[5/8] Verificando LangChain Google GenAI...")
try:
    import langchain_google_genai
    print(f"✓ LangChain Google GenAI instalado")
except ImportError:
    print("❌ LangChain Google GenAI no instalado. Ejecuta: pip install langchain-google-genai")
    sys.exit(1)

# ============================================================================
# 6. Verificar Google Generative AI
# ============================================================================
print("\n[6/8] Verificando Google Generative AI...")
try:
    import google.generativeai as genai
    print(f"✓ Google Generative AI instalado")
except ImportError:
    print("❌ Google Generative AI no instalado. Ejecuta: pip install google-generativeai")
    sys.exit(1)

# ============================================================================
# 6.5. Verificar kagglehub (opcional pero recomendado)
# ============================================================================
print("\n[6.5/8] Verificando kagglehub (para descarga automática de dataset)...")
try:
    import kagglehub
    print(f"✓ kagglehub instalado (descarga automática habilitada)")
except ImportError:
    print("⚠️  kagglehub no instalado (opcional)")
    print("    Para descarga automática: pip install kagglehub")
    print("    Alternativa: Descarga manual del dataset")

# ============================================================================
# 7. Verificar Artefactos del Modelo
# ============================================================================
print("\n[7/8] Verificando artefactos del modelo...")
required_files = ['modelo_depresion.h5', 'tokenizer.pickle', 'model_config.pickle']
alternative_dataset_names = ['reddit_depression_dataset.csv', 'depression_dataset.csv']
missing_files = []

for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file) / (1024 * 1024)  # MB
        print(f"✓ {file} ({size:.2f} MB)")
    else:
        missing_files.append(file)
        print(f"❌ {file} no encontrado")

if missing_files:
    print(f"\n⚠️  Archivos faltantes: {', '.join(missing_files)}")
    print("    Ejecuta train_model.py primero para generar estos archivos")
else:
    print("\n✓ Todos los artefactos del modelo están presentes")

# ============================================================================
# 8. Verificar API Key de Google
# ============================================================================
print("\n[8/8] Verificando Google API Key...")
api_key = os.environ.get('GOOGLE_API_KEY', '')

if not api_key or api_key == 'TU_API_KEY_AQUI':
    print("⚠️  GOOGLE_API_KEY no configurada")
    print("    Opciones:")
    print("    1. export GOOGLE_API_KEY='tu_api_key'")
    print("    2. Editar app.py línea 44")
    print("    3. Crear archivo .env")
else:
    print(f"✓ GOOGLE_API_KEY configurada (longitud: {len(api_key)} caracteres)")

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "=" * 80)
print("📊 RESUMEN DEL TEST")
print("=" * 80)

all_good = True

if python_version.major == 3 and python_version.minor >= 9:
    print("✓ Python")
else:
    print("❌ Python")
    all_good = False

try:
    import tensorflow
    print("✓ TensorFlow")
except:
    print("❌ TensorFlow")
    all_good = False

try:
    import streamlit
    print("✓ Streamlit")
except:
    print("❌ Streamlit")
    all_good = False

try:
    import crewai
    print("✓ CrewAI")
except:
    print("❌ CrewAI")
    all_good = False

try:
    import langchain_google_genai
    print("✓ LangChain Google")
except:
    print("❌ LangChain Google")
    all_good = False

if not missing_files:
    print("✓ Artefactos del modelo")
else:
    print("⚠️  Artefactos del modelo (parcialmente)")

if api_key and api_key != 'TU_API_KEY_AQUI':
    print("✓ Google API Key")
else:
    print("⚠️  Google API Key")

print("\n" + "=" * 80)
if all_good and not missing_files:
    print("🎉 ¡TODOS LOS COMPONENTES ESTÁN LISTOS!")
    print("=" * 80)
    print("\n📝 Próximos pasos:")
    print("   1. Si no tienes artefactos, ejecuta: python train_model.py")
    print("   2. Configura tu GOOGLE_API_KEY")
    print("   3. Ejecuta la app: streamlit run app.py")
else:
    print("⚠️  HAY COMPONENTES FALTANTES")
    print("=" * 80)
    print("\n📝 Acciones necesarias:")
    print("   1. Instala dependencias faltantes: pip install -r requirements.txt")
    if missing_files:
        print("   2. Entrena el modelo: python train_model.py")
    if not api_key or api_key == 'TU_API_KEY_AQUI':
        print("   3. Configura GOOGLE_API_KEY")

print("\n" + "=" * 80)
