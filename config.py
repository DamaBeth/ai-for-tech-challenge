import os
from dotenv import load_dotenv

# Para Windows - Utilizar en desarrollo, comentar en otro caso
import certifi
import warnings 
warnings.filterwarnings("ignore", category = DeprecationWarning)
#-------------------------------------------------------------

# Integración con Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- Configuración de variables de entorno ---
load_dotenv()

FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

# Para Windows - Utilizar en desarrollo, comentar en otro caso
# Forzar a httplib2/requests a usar el bundle de certificados de certifi
os.environ["HTTPLIB2_CA_CERTS"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()
#-------------------------------------------------------------


# --- Instanciación de Modelos ---

# Modelo para embeddings
# Nota: gemini-embedding-001 genera 3072 dimensiones (debe coincidir con el índice de Pinecone)
# En este caso sólo utilizaremos 768
embeddings_model = GoogleGenerativeAIEmbeddings(
  model = "gemini-embedding-001",
  output_dimensionality = 768,
  google_api_key = GOOGLE_GEMINI_API_KEY
)

# Modelo del LLM
llm = ChatGoogleGenerativeAI(
  api_key = GOOGLE_GEMINI_API_KEY,
  model = "gemini-2.5-flash", 
  temperature = 0.2
)