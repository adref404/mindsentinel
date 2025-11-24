"""
MindSentinel - Gestor de Base de Datos
======================================
Gestión de pacientes, evaluaciones y historial clínico
"""

import sqlite3
import hashlib
import secrets
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    """Gestor centralizado de la base de datos SQLite"""
    
    def __init__(self, db_path='database/mindsentinel.db'):
        """
        Inicializar conexión a base de datos
        
        Args:
            db_path: Ruta al archivo de base de datos
        """
        # Crear directorio si no existe
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establecer conexión con la base de datos"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Permite acceso por nombre de columna
        self.cursor = self.conn.cursor()
    
    def close(self):
        """Cerrar conexión a la base de datos"""
        if self.conn:
            self.conn.close()
    
    def initialize_database(self):
        """Crear tablas si no existen"""
        self.connect()
        
        # Tabla de pacientes
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS pacientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            codigo_paciente TEXT UNIQUE NOT NULL,
            nombre_completo TEXT NOT NULL,
            edad INTEGER,
            genero TEXT,
            email TEXT UNIQUE,
            telefono TEXT,
            password_hash TEXT NOT NULL,
            fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP,
            ultimo_acceso DATETIME,
            activo INTEGER DEFAULT 1
        )
        """)
        
        # Tabla de evaluaciones
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS evaluaciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paciente_id INTEGER NOT NULL,
            fecha_evaluacion DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            -- Datos del post analizado
            titulo_post TEXT NOT NULL,
            cuerpo_post TEXT NOT NULL,
            subreddit TEXT,
            
            -- Resultados del Agente Clasificador
            probabilidad_depresion REAL NOT NULL,
            nivel_riesgo TEXT NOT NULL,
            confianza_modelo REAL,
            prediccion_texto TEXT,
            
            -- Análisis XAI (Agente Explicador)
            analisis_xai TEXT,
            palabras_clave TEXT,
            patrones_linguisticos TEXT,
            distorsiones_cognitivas TEXT,
            tono_emocional TEXT,
            
            -- Decisión Supervisor (Agente Supervisor)
            decision_supervisor TEXT,
            recomendaciones TEXT,
            nivel_intervencion TEXT,
            requiere_seguimiento INTEGER DEFAULT 0,
            
            -- Metadatos
            duracion_analisis_segundos REAL,
            version_modelo TEXT,
            
            FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
        )
        """)
        
        # Tabla de sesiones (para autenticación)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS sesiones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paciente_id INTEGER NOT NULL,
            session_token TEXT UNIQUE NOT NULL,
            fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
            fecha_expiracion DATETIME NOT NULL,
            activa INTEGER DEFAULT 1,
            
            FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
        )
        """)
        
        # Tabla de notas clínicas (opcional, para médicos)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS notas_clinicas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluacion_id INTEGER NOT NULL,
            paciente_id INTEGER NOT NULL,
            profesional_nombre TEXT,
            nota TEXT NOT NULL,
            fecha_nota DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            FOREIGN KEY (evaluacion_id) REFERENCES evaluaciones(id),
            FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
        )
        """)
        
        # Índices para mejorar rendimiento
        self.cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_paciente_codigo 
        ON pacientes(codigo_paciente)
        """)
        
        self.cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_evaluaciones_paciente 
        ON evaluaciones(paciente_id, fecha_evaluacion DESC)
        """)
        
        self.conn.commit()
        print("✅ Base de datos inicializada correctamente")
    
    # =====================================================================
    # MÉTODOS DE PACIENTES
    # =====================================================================
    
    def hash_password(self, password):
        """Hash seguro de contraseña con SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def generar_codigo_paciente(self):
        """Generar código único de paciente (ej: MS-2024-0001)"""
        self.connect()
        year = datetime.now().year
        
        # Contar pacientes del año actual
        self.cursor.execute("""
        SELECT COUNT(*) as total FROM pacientes 
        WHERE codigo_paciente LIKE ?
        """, (f'MS-{year}-%',))
        
        total = self.cursor.fetchone()['total']
        nuevo_numero = total + 1
        
        return f"MS-{year}-{nuevo_numero:04d}"
    
    def registrar_paciente(self, nombre_completo, edad, genero, email, password, telefono=None):
        """
        Registrar nuevo paciente
        
        Returns:
            dict: Información del paciente registrado o None si falla
        """
        try:
            self.connect()
            
            # Generar código único
            codigo_paciente = self.generar_codigo_paciente()
            
            # Hash de contraseña
            password_hash = self.hash_password(password)
            
            # Insertar paciente
            self.cursor.execute("""
            INSERT INTO pacientes (
                codigo_paciente, nombre_completo, edad, genero, 
                email, telefono, password_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (codigo_paciente, nombre_completo, edad, genero, 
                  email, telefono, password_hash))
            
            self.conn.commit()
            paciente_id = self.cursor.lastrowid
            
            return {
                'id': paciente_id,
                'codigo_paciente': codigo_paciente,
                'nombre_completo': nombre_completo,
                'email': email
            }
        
        except sqlite3.IntegrityError as e:
            print(f"❌ Error: Email ya registrado - {e}")
            return None
        except Exception as e:
            print(f"❌ Error al registrar paciente: {e}")
            return None
    
    def autenticar_paciente(self, email, password):
        """
        Autenticar paciente con email y contraseña
        
        Returns:
            dict: Información del paciente o None si falla
        """
        try:
            self.connect()
            password_hash = self.hash_password(password)
            
            self.cursor.execute("""
            SELECT id, codigo_paciente, nombre_completo, email, edad, genero
            FROM pacientes
            WHERE email = ? AND password_hash = ? AND activo = 1
            """, (email, password_hash))
            
            paciente = self.cursor.fetchone()
            
            if paciente:
                # Actualizar último acceso
                self.cursor.execute("""
                UPDATE pacientes SET ultimo_acceso = CURRENT_TIMESTAMP
                WHERE id = ?
                """, (paciente['id'],))
                self.conn.commit()
                
                return dict(paciente)
            
            return None
        
        except Exception as e:
            print(f"❌ Error en autenticación: {e}")
            return None
    
    def obtener_paciente(self, paciente_id):
        """Obtener información completa del paciente"""
        self.connect()
        
        self.cursor.execute("""
        SELECT id, codigo_paciente, nombre_completo, edad, genero, 
               email, telefono, fecha_registro, ultimo_acceso
        FROM pacientes
        WHERE id = ? AND activo = 1
        """, (paciente_id,))
        
        paciente = self.cursor.fetchone()
        return dict(paciente) if paciente else None
    
    # =====================================================================
    # MÉTODOS DE EVALUACIONES
    # =====================================================================
    
    def guardar_evaluacion(self, paciente_id, datos_evaluacion):
        """
        Guardar evaluación completa del sistema multi-agente
        
        Args:
            paciente_id: ID del paciente
            datos_evaluacion: Diccionario con todos los resultados
        
        Returns:
            int: ID de la evaluación guardada
        """
        try:
            self.connect()
            
            self.cursor.execute("""
            INSERT INTO evaluaciones (
                paciente_id, titulo_post, cuerpo_post, subreddit,
                probabilidad_depresion, nivel_riesgo, confianza_modelo,
                prediccion_texto, analisis_xai, decision_supervisor,
                recomendaciones, nivel_intervencion, duracion_analisis_segundos
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                paciente_id,
                datos_evaluacion.get('titulo_post'),
                datos_evaluacion.get('cuerpo_post'),
                datos_evaluacion.get('subreddit'),
                datos_evaluacion.get('probabilidad_depresion'),
                datos_evaluacion.get('nivel_riesgo'),
                datos_evaluacion.get('confianza_modelo'),
                datos_evaluacion.get('prediccion_texto'),
                datos_evaluacion.get('analisis_xai'),
                datos_evaluacion.get('decision_supervisor'),
                datos_evaluacion.get('recomendaciones'),
                datos_evaluacion.get('nivel_intervencion'),
                datos_evaluacion.get('duracion_analisis_segundos', 0)
            ))
            
            self.conn.commit()
            evaluacion_id = self.cursor.lastrowid
            
            print(f"✅ Evaluación guardada con ID: {evaluacion_id}")
            return evaluacion_id
        
        except Exception as e:
            print(f"❌ Error al guardar evaluación: {e}")
            return None
    
    def obtener_evaluaciones_paciente(self, paciente_id, limite=10):
        """
        Obtener historial de evaluaciones de un paciente
        
        Args:
            paciente_id: ID del paciente
            limite: Número máximo de evaluaciones a retornar
        
        Returns:
            list: Lista de evaluaciones ordenadas por fecha (más recientes primero)
        """
        self.connect()
        
        self.cursor.execute("""
        SELECT 
            id, fecha_evaluacion, titulo_post, 
            probabilidad_depresion, nivel_riesgo, nivel_intervencion
        FROM evaluaciones
        WHERE paciente_id = ?
        ORDER BY fecha_evaluacion DESC
        LIMIT ?
        """, (paciente_id, limite))
        
        evaluaciones = [dict(row) for row in self.cursor.fetchall()]
        return evaluaciones
    
    def obtener_evaluacion_completa(self, evaluacion_id):
        """Obtener todos los detalles de una evaluación específica"""
        self.connect()
        
        self.cursor.execute("""
        SELECT * FROM evaluaciones
        WHERE id = ?
        """, (evaluacion_id,))
        
        evaluacion = self.cursor.fetchone()
        return dict(evaluacion) if evaluacion else None
    
    def obtener_estadisticas_paciente(self, paciente_id):
        """
        Calcular estadísticas del historial del paciente
        
        Returns:
            dict: Estadísticas agregadas
        """
        self.connect()
        
        self.cursor.execute("""
        SELECT 
            COUNT(*) as total_evaluaciones,
            AVG(probabilidad_depresion) as promedio_probabilidad,
            MAX(probabilidad_depresion) as maxima_probabilidad,
            MIN(probabilidad_depresion) as minima_probabilidad,
            SUM(CASE WHEN nivel_riesgo = 'ALTO' THEN 1 ELSE 0 END) as evaluaciones_alto_riesgo,
            SUM(CASE WHEN nivel_riesgo = 'MEDIO' THEN 1 ELSE 0 END) as evaluaciones_medio_riesgo,
            SUM(CASE WHEN nivel_riesgo = 'BAJO' THEN 1 ELSE 0 END) as evaluaciones_bajo_riesgo
        FROM evaluaciones
        WHERE paciente_id = ?
        """, (paciente_id,))
        
        stats = self.cursor.fetchone()
        return dict(stats) if stats else None
    
    # =====================================================================
    # MÉTODOS DE SESIONES
    # =====================================================================
    
    def crear_sesion(self, paciente_id):
        """Crear sesión de autenticación para el paciente"""
        self.connect()
        
        # Generar token único
        session_token = secrets.token_urlsafe(32)
        
        # Expiración en 7 días
        from datetime import timedelta
        fecha_expiracion = datetime.now() + timedelta(days=7)
        
        self.cursor.execute("""
        INSERT INTO sesiones (paciente_id, session_token, fecha_expiracion)
        VALUES (?, ?, ?)
        """, (paciente_id, session_token, fecha_expiracion))
        
        self.conn.commit()
        return session_token
    
    def validar_sesion(self, session_token):
        """Validar si una sesión es válida y activa"""
        self.connect()
        
        self.cursor.execute("""
        SELECT paciente_id FROM sesiones
        WHERE session_token = ? 
        AND activa = 1 
        AND fecha_expiracion > CURRENT_TIMESTAMP
        """, (session_token,))
        
        sesion = self.cursor.fetchone()
        return sesion['paciente_id'] if sesion else None
    
    def cerrar_sesion(self, session_token):
        """Cerrar sesión del paciente"""
        self.connect()
        
        self.cursor.execute("""
        UPDATE sesiones SET activa = 0
        WHERE session_token = ?
        """, (session_token,))
        
        self.conn.commit()


# Instancia global del gestor de base de datos
db = DatabaseManager()
