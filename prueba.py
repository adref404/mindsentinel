import sys

print("🔍 Verificando instalación...")

# 1. Verificar CrewAI
try:
    from crewai import LLM
    print("✅ CrewAI instalado")
except ImportError as e:
    print(f"❌ CrewAI no instalado: {e}")
    sys.exit(1)

# 2. Verificar Google Generative AI
try:
    import google.generativeai as genai
    print("✅ google-generativeai instalado")
except ImportError as e:
    print(f"❌ google-generativeai no instalado: {e}")
    sys.exit(1)

# 3. Verificar LiteLLM
try:
    import litellm
    print("✅ litellm instalado")
except ImportError as e:
    print(f"❌ litellm no instalado: {e}")
    sys.exit(1)

# 4. Verificar .env
try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        print(f"✅ GOOGLE_API_KEY encontrada: {api_key[:10]}...")
    else:
        print("⚠️ GOOGLE_API_KEY no encontrada en .env")
except Exception as e:
    print(f"❌ Error con .env: {e}")

# 5. Test de conexión con Gemini
try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️ No se puede probar conexión sin API Key")
    else:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Di 'OK' si funcionas")
        print(f"✅ Conexión con Gemini exitosa: {response.text[:50]}")
except Exception as e:
    print(f"❌ Error al conectar con Gemini: {e}")

# 6. Test de CrewAI LLM
try:
    from crewai import LLM
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=api_key,
            temperature=0.7
        )
        print("✅ LLM de CrewAI configurado correctamente")
    else:
        print("⚠️ No se puede crear LLM sin API Key")
except Exception as e:
    print(f"❌ Error al configurar LLM de CrewAI: {e}")

print("\n" + "="*50)
print("✨ Verificación completada")
print("="*50)