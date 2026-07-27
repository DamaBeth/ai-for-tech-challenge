import sqlite3
import streamlit as st
from embeddings import PineconeService
from zorawaru_agent import ask_agent  # Importa tu función del agente
from langchain_core.messages import HumanMessage, AIMessage

# -------------------------------------------------------------------
# 1. EJECUCIÓN ÚNICA Y FUNCIONES DE BASE DE DATOS
# -------------------------------------------------------------------

@st.cache_resource
def init_data_base():
    """Se ejecuta una sola vez al cargar la aplicación por primera vez en el servidor."""
    with st.spinner("🔄 Sincronizando base de datos vectorial con Google Drive..."):
        sync_service = PineconeService()
        sync_service.sync()
    return True


def init_sqlite_db():
    """Crea la tabla de feedback si no existe."""
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pregunta TEXT,
            respuesta TEXT,
            tipo_feedback TEXT,  -- 'positivo' o 'negativo'
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_feedback(pregunta: str, respuesta: str, tipo: str):
    """Inserta una valoración en la base de datos de manera silenciosa."""
    conn = sqlite3.connect("feedback.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO feedback (pregunta, respuesta, tipo_feedback) VALUES (?, ?, ?)",
        (pregunta, respuesta, tipo)
    )
    conn.commit()
    conn.close()


# Llamamos a las inicializaciones antes de construir la interfaz
init_data_base()    # Sincroniza Pinecone (cacheado)
init_sqlite_db()   # Asegura que exista feedback.db

# -------------------------------------------------------------------
# 2. CONFIGURACIÓN DE LA INTERFAZ Y ESTADO DE SESIÓN
# -------------------------------------------------------------------
st.set_page_config(page_title="Zorawaru - Senkats", page_icon="🦊")
st.title("🦊 Zorawaru - Asistente de Senkats")

# Inicializar el historial de chat
if "chat_history" not in st.session_state:
    welcome_message = (
        "¡Hola! 🦊 Soy Zorawaru, tu asistente virtual experto en la plataforma Senkats. "
        "¿En qué te puedo ayudar el día de hoy?"
    )
    st.session_state.chat_history = [
        AIMessage(content=welcome_message)
    ]

# Registro auxiliar de feedbacks dados en la sesión actual para evitar duplicados
if "feedbacks_registrados" not in st.session_state:
    st.session_state.feedbacks_registrados = set()

# -------------------------------------------------------------------
# 3. INTERFAZ DE CHAT Y LÓGICA DE MENSAJES
# -------------------------------------------------------------------
# Renderizar el historial de conversación
for idx, msg in enumerate(st.session_state.chat_history):
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    
    with st.chat_message(role):
        st.markdown(msg.content, unsafe_allow_html=True)
        
        # Mostrar botones de pulgares ÚNICAMENTE en las respuestas del asistente (excepto el saludo inicial)
        if isinstance(msg, AIMessage) and idx > 0:
            feedback_key = f"feedback_{idx}"
            
            # Si el usuario no ha votado esta respuesta
            if idx not in st.session_state.feedbacks_registrados:
                opcion = st.feedback("thumbs", key=feedback_key)
                
                if opcion is not None:
                    # 0 = Pulgar abajo (negativo), 1 = Pulgar arriba (positivo)
                    tipo_feedback = "positivo" if opcion == 1 else "negativo"
                    
                    # Recuperar la pregunta del usuario anterior para guardar el contexto completo
                    pregunta_relacionada = (
                        st.session_state.chat_history[idx - 1].content 
                        if idx - 1 >= 0 else "N/A"
                    )
                    
                    # Almacenar en la base de datos sin mostrar estadísticas
                    save_feedback(pregunta_relacionada, msg.content, tipo_feedback)
                    st.session_state.feedbacks_registrados.add(idx)
                    
                    st.toast("¡Gracias por tu valoración! 🦊", icon="🎉")
                    st.rerun()
            else:
                st.caption("✅ Valoración registrada")

# Capturar entrada del usuario
user_message = st.chat_input("Escribe tu pregunta sobre Senkats...")

if user_message:
    # 1. Mostrar inmediatamente la pregunta del usuario
    with st.chat_message("user"):
        st.markdown(user_message)

    # 2. Consultar al agente manteniendo el contexto
    with st.spinner("Zorawaru está consultando la información... 🦊"):
        zorawaru_response = ask_agent(user_message, st.session_state.chat_history)

    # 3. Mostrar la respuesta de Zorawaru de inmediato en pantalla
    with st.chat_message("assistant"):
        st.markdown(zorawaru_response, unsafe_allow_html=True)

    # 4. Guardar en el historial de sesión
    st.session_state.chat_history.append(HumanMessage(content=user_message))
    st.session_state.chat_history.append(AIMessage(content=zorawaru_response))

    # 5. Rerun para adjuntar dinámicamente los botones de feedback al nuevo mensaje
    st.rerun()