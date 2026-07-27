import json
# Google Service Account Credentials
from google.oauth2.service_account import Credentials
# Variables de entorno
from config import GOOGLE_SERVICE_ACCOUNT_JSON

def get_Google_Drive_credentials():
    # Genera credenciales de la cuenta de servicio de Google Cloud a partir de la variable de entorno.
    json_str = GOOGLE_SERVICE_ACCOUNT_JSON
    if not json_str:
        raise ValueError("Falta la variable GOOGLE_SERVICE_ACCOUNT_JSON en el archivo .env")
    
    account_info = json.loads(json_str)
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    drive_credentials = Credentials.from_service_account_info(account_info, scopes=scopes)
    return drive_credentials