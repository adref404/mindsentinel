# 🧠 MindSentinel - Resumen Ejecutivo del Proyecto

## 📌 Información General

| Campo | Detalle |
|-------|---------|
| **Nombre** | MindSentinel |
| **Tipo** | Sistema Multi-Agente para Monitoreo de Salud Mental |
| **Objetivo** | Detectar indicadores de depresión en texto de redes sociales (Reddit) |
| **Nivel** | Proyecto Académico / Investigación |
| **Tecnologías** | TensorFlow, CrewAI, Google Gemini, Streamlit |

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

## 📂 Archivos del Proyecto

### 📄 Scripts Principales

1. **`train_model.py`** (16 KB)
   - Script de entrenamiento del modelo LSTM
   - Genera artefactos: modelo.h5, tokenizer.pickle, config.pickle
   - Ejecutar: `python train_model.py`

2. **`app.py`** (23 KB)
   - Aplicación web con Streamlit
   - Orquesta los 3 agentes con CrewAI
   - Ejecutar: `streamlit run app.py`

### 📄 Archivos de Soporte

3. **`requirements.txt`** (449 bytes)
   - Todas las dependencias del proyecto
   - Instalar: `pip install -r requirements.txt`

4. **`test_system.py`** (6.2 KB)
   - Verifica que todos los componentes estén instalados
   - Ejecutar: `python test_system.py`

5. **`README.md`** (9.6 KB)
   - Documentación completa del proyecto
   - Guía de instalación y uso

6. **`.env.example`** (470 bytes)
   - Plantilla para variables de entorno
   - Configuración de GOOGLE_API_KEY

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

## 📖 Publicaciones Relacionadas

Este proyecto se basa en investigaciones como:

1. **Coppersmith et al. (2015)** - "Quantifying Mental Health Signals in Twitter"
2. **Hochreiter & Schmidhuber (1997)** - "Long Short-Term Memory"
3. **Wooldridge & Jennings (1995)** - "Intelligent Agents"
4. **Holzinger et al. (2022)** - "Explainable AI in Healthcare"

---

## ⚠️ Consideraciones Éticas

### ❌ NO debe usarse para:
- Diagnóstico clínico real
- Sustitución de terapia profesional
- Decisiones médicas sin supervisión
- Vigilancia no consentida

### ✅ SÍ puede usarse para:
- Investigación académica controlada
- Desarrollo de herramientas de detección temprana
- Estudios de viabilidad técnica
- Educación en IA y salud mental

---

## 🔐 Seguridad y Privacidad

- **Datos**: Solo texto simulado, sin información personal real
- **API Keys**: Nunca compartir GOOGLE_API_KEY públicamente
- **Almacenamiento**: Los modelos se guardan localmente
- **GDPR/HIPAA**: No aplicable (proyecto académico sin datos reales)

---

## 🛠️ Requisitos del Sistema

### Hardware Mínimo
- CPU: Intel Core i5 o equivalente
- RAM: 8 GB
- Disco: 2 GB libres

### Hardware Recomendado (Entrenamiento)
- GPU: NVIDIA con CUDA (opcional, acelera 10x)
- RAM: 16 GB
- Disco: 5 GB libres

### Software
- Python 3.9 o superior
- pip (gestor de paquetes)
- Navegador web moderno

---

## 📞 Recursos de Ayuda

### Líneas de Crisis (en caso real)
- pe Perú: **0800-10828** (Infosalud )
- 🇲🇽 México: **800 290 0024** (SAPTEL)
- 🇦🇷 Argentina: **135** (Centro de Asistencia al Suicida)
- 🇺🇸 USA: **988** (Suicide & Crisis Lifeline)
- 🌍 Internacional: [findahelpline.com](https://findahelpline.com)

### Enlaces Útiles
- [Google AI Studio](https://makersuite.google.com/app/apikey) - API Key gratuita
- [Kaggle Dataset](https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media)
- [CrewAI Docs](https://docs.crewai.com)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

---

## 📈 Roadmap Futuro

### Posibles Mejoras:
- [ ] Integración con BERT o GPT para mejor precisión
- [ ] Análisis multimodal (texto + imágenes)
- [ ] Dashboard de monitoreo temporal
- [ ] API REST para integración con otras apps
- [ ] Soporte para más idiomas (actualmente: español e inglés)
- [ ] Detección de otras condiciones (ansiedad, PTSD)

---

## 🤝 Contribuciones

Este es un proyecto académico abierto. Para contribuir:
1. Fork el repositorio
2. Crea una rama feature (`git checkout -b feature/mejora`)
3. Commit tus cambios (`git commit -m 'Agregar mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

---

## 📜 Licencia

Uso académico e investigación. No usar en producción médica sin validación clínica.

---

## 👨‍💻 Créditos

**Desarrollado para:** Proyecto Académico de IA y Salud Mental

**Tecnologías:**
- TensorFlow/Keras (Deep Learning)
- CrewAI (Orquestación de agentes)
- Google Gemini (LLM)
- Streamlit (Frontend)
- LangChain (Integración LLM)

**Dataset:** Reddit Mental Health Social Media (Kaggle)

---

## 🎉 ¡Comienza Ahora!

1. **Descarga** todos los archivos del proyecto
2. **Sigue** las instrucciones
3. **Entrena** el modelo con `train_model.py`
4. **Ejecuta** la aplicación con `streamlit run app.py`
5. **Explora** y mejora el sistema

---

**🧠 MindSentinel** - Arquitectura Multi-Agente para el Monitoreo de Salud Mental

*"Inteligencia Artificial al servicio del bienestar humano"*

---

**Última actualización:** Noviembre 2025
**Versión:** 1.0
