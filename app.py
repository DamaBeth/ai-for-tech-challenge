import streamlit as st
from embeddings import PineconeService
from zorawaru_agent import ask_agent  # Importa tu función del agente
from langchain_core.messages import HumanMessage, AIMessage

# -------------------------------------------------------------------
# 1. EJECUCIÓN ÚNICA AL INICIAR LA APLICACIÓN
# -------------------------------------------------------------------
@st.cache_resource
def init_data_base():
    """Se ejecuta una sola vez al cargar la aplicación por primera vez en el servidor."""
    st.write("🔄 Sincronizando base de datos vectorial...")
    sync_service = PineconeService()
    sync_service.sync()
    return True

# Llamamos a la función de inicialización antes de construir la interfaz
init_data_base()


# -------------------------------------------------------------------
# 2. CONFIGURACIÓN DE LA INTERFAZ Y ESTADO DE SESIÓN
# -------------------------------------------------------------------
st.set_page_config(page_title="Zorawaru - Senkats", page_icon="🦊")
st.title("🦊 Zorawaru - Asistente de Senkats")

# Inicializar el historial de chat si es una sesión nueva de usuario
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------------------------------------------------
# 3. INTERFAZ DE CHAT Y LÓGICA DE MENSAJES
# -------------------------------------------------------------------
# Renderizar mensajes del historial existente
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content, unsafe_allow_html=True)

# Capturar entrada del usuario
user_message = st.chat_input("Escribe tu pregunta sobre nuestra plataforma educativa Senkats...")

if user_message:
    # Mostrar la pregunta del usuario
    with st.chat_message("user"):
        st.markdown(user_message)

    # Consultar al agente manteniendo el historial de esta sesión
    with st.spinner("Zorawaru está consultando la información... 🦊"):
        zorawaru_response = ask_agent(user_message, st.session_state.chat_history)

    # Mostrar la respuesta del agente
    with st.chat_message("assistant"):
        st.markdown(zorawaru_response, unsafe_allow_html=True)

    # Guardar en el historial
    st.session_state.chat_history.append(HumanMessage(content=user_message))
    st.session_state.chat_history.append(AIMessage(content=zorawaru_response))