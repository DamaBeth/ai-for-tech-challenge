# Senkats

Challenge Alura - Zorawaru, Agente conversacional para la plataforma educativa Senkats.

## 🧩 Descripción del proyecto

Zorawaru es un asistente virtual basado en inteligencia artificial que permite consultar información de la plataforma educativa ficticia Senkats mediante conversaciones naturales. El proyecto implementa un flujo RAG (Retrieval-Augmented Generation) que recupera documentos relevantes desde Google Drive, los transforma en embeddings y los almacena en Pinecone para responder preguntas con contexto real y actualizado.

La aplicación cuenta con una interfaz web construida con Streamlit, donde el usuario puede hacer preguntas sobre la plataforma y recibir respuestas generadas por Google Gemini, además de registrar feedback sobre la calidad de las respuestas.

## 🏗️ Arquitectura del sistema

El flujo del proyecto sigue esta lógica:

1. Los documentos de referencia se almacenan en una carpeta de Google Drive.
2. El servicio de sincronización accede a esos archivos PDF y extrae su contenido.
3. Los textos se dividen en fragmentos y se generan embeddings con Google Gemini.
4. Los embeddings se almacenan en un índice vectorial de Pinecone.
5. Cuando el usuario realiza una pregunta, la aplicación recupera los fragmentos más relevantes y los envía al modelo de lenguaje junto con el historial de conversación.
6. La respuesta se muestra en la interfaz de Streamlit y se guarda una valoración de feedback en SQLite.

### Archivos principales

- app.py: interfaz de usuario con Streamlit y lógica de chat.
- zorawaru_agent.py: orquestación del flujo RAG y generación de respuestas.
- embeddings.py: sincronización, indexación y actualización de documentos en Pinecone.
- config.py: configuración de modelos y variables de entorno.
- get_google_credentials.py: autenticación con Google Drive mediante cuenta de servicio.

## 🛠️ Tecnologías utilizadas

- Python 3
- Streamlit para la interfaz de usuario
- LangChain para el flujo de prompts y recuperación de contexto
- Google Gemini para embeddings y generación de respuestas
- Pinecone como base de datos vectorial
- Google Drive API para lectura de documentos PDF
- SQLite para almacenar feedback de los usuarios
- python-dotenv para cargar variables de entorno

## 🧠 Base de conocimiento - Documentación de Senkats

Puedes consultar la documentación original de la base de conocimiento en Google Drive a través del siguiente enlace:

https://drive.google.com/drive/folders/1MJHsNQuFzdCDbSVOvCDTTz40F5lxAXyA?usp=sharing

## ▶️ Instrucciones de ejecución

### 1. Activar entorno virtual

Crea y activa un entorno virtual en tu máquina:

  - venv en Windows:

    ```bash
    python -m venv .venv-gemini-v2
    .\.venv-gemini-v2\Scripts\activate
    ```
  - venv en Mac/Linux:

    ```bash
    python3 -m venv .venv-gemini-v2
    source .venv-gemini-v2/bin/activate
    ```

### 2. Instalar dependencias

Instala las librerías necesarias con:

```bash
pip install -r requirements.txt
```

### 3. Preparar la base de conocimiento

- Accede a la carpeta de Google Drive compartida con la documentación de Senkats.
- Descarga los archivos PDF que deseas que el agente pueda consultar.
- Crea una carpeta nueva en tu Google Drive si es necesario.
- Sube allí los documentos que formarán parte del conocimiento del asistente.

### 4. Obtener credenciales de Google Drive

1. Crea un proyecto en Google Cloud.
2. Habilita la API de Google Drive.
3. Genera una cuenta de servicio y descarga el archivo JSON de credenciales.
4. Comparte la carpeta de Google Drive con el correo de la cuenta de servicio (sólo permisos de lectura).

### 5. Crear una API de Google Gemini

1. Inicia sesión en Google AI Studio o en la consola de Google Cloud.
2. Genera una API key para Gemini.
3. Guarda la clave de forma segura.

### 6. Crear una cuenta en Pinecone

1. Regístrate en Pinecone.
2. Crea una API key desde la sección de Keys.
3. Crea un índice vectorial personalizado (presiona Custom Settings) con las siguientes características:
   - Nombre: drive-rag-index
   - Dimensiones: 768 (debe de coindicir con las dimensiones del modelo)
   - Métrica: cosine
   - Tipo de vector: dense
   - Modo de capacidad: serverless

### 7. Configurar variables de entorno

Crea un archivo .env con las siguientes variables:

```env
GOOGLE_GEMINI_API_KEY="TU_API_KEY_AQUÍ"
GOOGLE_DRIVE_FOLDER_ID="tu-folder-id-de-drive-de-base-de-conocimiento"
GOOGLE_SERVICE_ACCOUNT_JSON='{"type": "service_account", "project_id": "...", "private_key_id": "...", "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n", "client_email": "...", "client_id": "...", "auth_uri": "...", "token_uri": "...", "auth_provider_x509_cert_url": "...", "client_x509_cert_url": "..."}'
PINECONE_API_KEY="tu-pinecone-api-key"
PINECONE_INDEX_NAME="drive-rag-index"
```

### 8. Ejecutar la aplicación

Desde la raíz del proyecto, ejecuta:

```bash
streamlit run app.py
```

## 💬 Ejemplos de preguntas y respuestas

### Inicio de la conversación

![Bienvenida y primera pregunta](assets/pregunta_1.png)

### Historial de conversación e indicativo de espera

![Indicador de espera](assets/historial.png)

### Mostrar fuentes consultadas

![Citación de fuentes](assets/pregunta_2.png)

### Ejemplo de una respuesta a pregunta fuera de contexto

![Respuesta a una pregunta fuera de contexto](assets/pregunta_fuera_de_contexto.png)

### Ejemplo de mensaje luego de indicar la valoración a una respuesta

![Respuesta valorada](assets/valorar_respuesta.png)

## 🌐 Despliegue

Puedes encontrar el demo del agente en el siguiente enlace:

- [Agregar URL de despliegue aquí](#)

### Registro de Ejecución en la Nube
![Logs de ejecución en producción]()
![Interactuando con la interfaz en la nube]()

## 📝 Notas adicionales

El proyecto está pensado para funcionar como un asistente de soporte y consulta para usuarios de Senkats, utilizando un enfoque moderno de búsqueda semántica y generación de respuestas demilitado a su contexto (documentación proporcionada).