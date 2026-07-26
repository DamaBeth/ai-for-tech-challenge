# Senkats

Challenge Alura - Agente de la plataforma educativa "Senkats"

## 🔨 Funcionalidades del proyecto

## 🧠 Base de conocimiento - Documentación de Senkats

Podrás encontrar los archivos originales de la base de conocimiento disponibles en Google Drive a través del enlace:

https://drive.google.com/drive/folders/1MJHsNQuFzdCDbSVOvCDTTz40F5lxAXyA?usp=sharing

## ✔️ Técnicas y tecnologías utilizadas

## 🛠️ Ejecuta el proyecto

### Paso 1 - 🤖 Activar entorno virtual (escoger mejor titulo)

### Paso 2 - ⬇️ Instalar dependencias

Instala los paquetes utilizando:

```bash
pip install -r requirements.txt
```

### Paso 2 - Descarga los archivos de la base de conocimiento

- Ingresa al enlace proporcionado de la base de conocimiento, ahí encontrarás los archivos en formato PDF que deberás descargar. 
- Descarga de los archivos
- Abre tu cuenta de Google Drive
- Crea una carpeta nueva para almacenar estos archivos
- Sube los archivos

### Paso 3 - Obtén las credenciales de acceso de Google Drive

### Paso 4 - Crea una API de Google Gemini

### Paso 5 - Crea una cuenta en Pinecone

  1. Ve a pinecone.io y regístrate gratis.
  2. Una vez que te hayas registrado, en el panel lateral, ve a API Keys
  3. Haz clic en _Create API Key_, asígnale un nombre y copia el token generado.
  4. Obtén / Verifica las dimensiones de tus Embeddings:
      En este caso el modelo de Google: _text-embedding-004_, genera vectores de 768 dimensiones
  5. Crea el Índice en Pinecone:

      - Selecciona la opción Database del menú lateral
      - Selecciona Indexes
      - Selecciona el botón Create Index e ingresa los siguientes datos:
      
        1. Ingresa el nombre del índice, por ejmeplo: drive-rag-index (o el nombre que prefieras).
        2. Selecciona la casilla Custom Settings (para el modelo de Google _text-embedding-004_) e ingresa los siguientes valores:
            - Dimensions: 768 (Importante: debe coincidir exactamente con el output del modelo).
            - Metric: cosine (es la recomendada para texto/embeddings).
            - Vector Type: Dense (es el formato estándar utilizado por prácticamente todos los modelos de embeddings)
        3. En el apartado Capacity Mode, selecciona: Serverless.
        4. En el apartado Cloud Provider & Region, selecciona: AWS y en la región us-east-1 (es la opción con nivel gratuito).

### Paso 6 - 🔑 Generar API\_KEYs y asociarlas al archivo .env

Crea un archivo .env en el que almacenarás las siguientes variables de entorno. Reemplza los valores por los que corresponden a tus credenciales.

```python
GOOGLE_GEMINI_API_KEY = "TU_API_KEY_AQUÍ"
# Lo encuentras al final de la URL de la carpeta donde almacenaste los archivos de la base de conocimiento (Paso 2)
GOOGLE_DRIVE_FOLDER_ID="tu-folder-id-de-drive-de-base-de-conocimiento"
# Pega el contenido completo de tu archivo JSON en una sola línea
GOOGLE_SERVICE_ACCOUNT_JSON='{"type": "service_account", "project_id": "...", "private_key_id": "...", "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n", "client_email": "...", "client_id": "...", "auth_uri": "...", "token_uri": "...", "auth_provider_x509_cert_url": "...", "client_x509_cert_url": "..."}'
# Variables de Pinecone
PINECONE_API_KEY="tu-pinecone-api-key"
PINECONE_INDEX_NAME="drive-rag-index"
```
### Paso 7 - 🚀 Ejecuta el proyecto

Abre la terminal en el IDE donde abriste el poryecto previamente y ejecuta el siguiente comando:

```bash
python main.py
```
