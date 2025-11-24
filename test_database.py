"""
Script de prueba para verificar funcionamiento de la base de datos
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from database.db_manager import DatabaseManager

def test_database():
    """Probar operaciones básicas de la base de datos"""
    
    print("=" * 60)
    print("TEST DE BASE DE DATOS - MindSentinel")
    print("=" * 60)
    
    # Inicializar base de datos
    print("\n1. Inicializando base de datos...")
    db = DatabaseManager('database/test_mindsentinel.db')
    db.initialize_database()
    print("✅ Base de datos inicializada")
    
    # Registrar paciente
    print("\n2. Registrando paciente de prueba...")
    paciente = db.registrar_paciente(
        nombre_completo="Juan Pérez Test",
        edad=25,
        genero="Masculino",
        email="juan.test@example.com",
        password="password123",
        telefono="+51 999 999 999"
    )
    
    if paciente:
        print(f"✅ Paciente registrado:")
        print(f"   - ID: {paciente['id']}")
        print(f"   - Código: {paciente['codigo_paciente']}")
        print(f"   - Nombre: {paciente['nombre_completo']}")
        print(f"   - Email: {paciente['email']}")
    else:
        print("❌ Error al registrar paciente")
        return
    
    # Autenticar paciente
    print("\n3. Autenticando paciente...")
    paciente_auth = db.autenticar_paciente("juan.test@example.com", "password123")
    
    if paciente_auth:
        print(f"✅ Autenticación exitosa")
        print(f"   - Bienvenido: {paciente_auth['nombre_completo']}")
    else:
        print("❌ Error en autenticación")
        return
    
    # Guardar evaluación de prueba
    print("\n4. Guardando evaluación de prueba...")
    datos_evaluacion = {
        'titulo_post': "Test post",
        'cuerpo_post': "Este es un post de prueba para verificar el sistema",
        'subreddit': "r/test",
        'probabilidad_depresion': 0.65,
        'nivel_riesgo': "MEDIO",
        'confianza_modelo': 0.75,
        'prediccion_texto': "Señales moderadas de riesgo",
        'analisis_xai': "Análisis de prueba XAI",
        'decision_supervisor': "Recomendación de monitoreo",
        'recomendaciones': "Seguimiento cercano recomendado",
        'nivel_intervencion': "MEDIO",
        'duracion_analisis_segundos': 5.5
    }
    
    evaluacion_id = db.guardar_evaluacion(paciente['id'], datos_evaluacion)
    
    if evaluacion_id:
        print(f"✅ Evaluación guardada con ID: {evaluacion_id}")
    else:
        print("❌ Error al guardar evaluación")
        return
    
    # Obtener historial
    print("\n5. Obteniendo historial de evaluaciones...")
    evaluaciones = db.obtener_evaluaciones_paciente(paciente['id'], limite=10)
    
    print(f"✅ Se encontraron {len(evaluaciones)} evaluaciones")
    for eval in evaluaciones:
        print(f"   - Evaluación {eval['id']}: {eval['titulo_post']} - Riesgo: {eval['nivel_riesgo']}")
    
    # Obtener estadísticas
    print("\n6. Obteniendo estadísticas del paciente...")
    stats = db.obtener_estadisticas_paciente(paciente['id'])
    
    if stats:
        print(f"✅ Estadísticas:")
        print(f"   - Total evaluaciones: {stats['total_evaluaciones']}")
        print(f"   - Probabilidad promedio: {stats['promedio_probabilidad']:.2%}")
        print(f"   - Riesgo alto: {stats['evaluaciones_alto_riesgo']}")
        print(f"   - Riesgo medio: {stats['evaluaciones_medio_riesgo']}")
        print(f"   - Riesgo bajo: {stats['evaluaciones_bajo_riesgo']}")
    
    print("\n" + "=" * 60)
    print("✅ TODOS LOS TESTS PASARON CORRECTAMENTE")
    print("=" * 60)
    
    # Limpiar base de datos de prueba
    print("\n⚠️  Limpiando base de datos de prueba...")
    import os
    if os.path.exists('database/test_mindsentinel.db'):
        os.remove('database/test_mindsentinel.db')
        print("✅ Base de datos de prueba eliminada")

if __name__ == "__main__":
    try:
        test_database()
    except Exception as e:
        print(f"\n❌ ERROR DURANTE LOS TESTS: {str(e)}")
        import traceback
        traceback.print_exc()
