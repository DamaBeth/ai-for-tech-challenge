# Google Service Account Credentials
from googleapiclient.discovery import build
# Carga de archivos desde Google Drive
from langchain_google_community import GoogleDriveLoader
# División de texto
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Integración con Pinecone para Almacenamiento de vectores (embeddings)
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

from config import embeddings_model, PINECONE_API_KEY, INDEX_NAME, FOLDER_ID
from get_google_credentials import get_Google_Drive_credentials

class PineconeService:
    def __init__(self, model=embeddings_model, folder_id=FOLDER_ID, index_name=INDEX_NAME, api_key=PINECONE_API_KEY):
        self.model = model
        self.folder_id = folder_id
        self.index_name = index_name
        self.google_credentials = get_Google_Drive_credentials()
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

    def sync(self):
        """Coordinador principal de sincronización."""
        if not self._has_index_vectors():
            print("🚀 Primera vez: Cargando e indexando documentos desde Google Drive por primera vez...")
            self.create_embeddings()
        else:
            print("🔄 Sincronizando cambios...")
            self.update_embeddings()

    def create_embeddings(self):
        """
          ** Indexación desde cero.** 
          Descarga los PDFs de Drive por primera vez, guarda metadatos clave (file_id, id y modifiedTime)
          e indexa de forma limpia y garantizada en Pinecone.
        """

        # 1. Obtener un mapa rápido de fechas de modificación y nombres desde la API de Drive
        service = build('drive', 'v3', credentials = self.google_credentials)
        results = service.files().list(
            q=f"'{self.folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            fields="files(id, name, modifiedTime)"
        ).execute()
    
        files_list = results.get('files', [])
        # Diccionario de fechas: {'id_del_archivo': '2026-07-24T00:00:00Z'}
        dates_map = {f['id']: f['modifiedTime'] for f in files_list}

        # 2. Cargar los documentos con el loader
        loader = GoogleDriveLoader(
            folder_id = self.folder_id,
            credentials = self.google_credentials,
            recursive = False,  # Ignora subcarpetas
            file_types = ["application/pdf"]
        )
    
        documents = loader.load()
        print(f"Se encontraron {len(documents)} archivos PDF.")

        if not documents:
            print("No se encontraron documentos en la raíz del folder.")
            return

        # 3. Dividir documentos completos en fragmentos (chunks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        print(f"Generando embeddings para {len(chunks)} fragmentos...")

        # 4. Enriquecer e inyectar metadatos explícitos en CADA chunk
        ids = []
        for i, chunk in enumerate(chunks):
            # Intentar obtener el ID desde las claves estándar de GoogleDriveLoader
            file_id = chunk.metadata.get("id") or chunk.metadata.get("file_id")

            # Si el loader guardó la ruta en 'source' o 'title' en vez del 'id', realizamos un fall-back
            if not file_id or file_id == "desconocido":
                source_text = chunk.metadata.get("source", "") or chunk.metadata.get("title", "")
                for f in files_list:
                    if f['id'] in source_text or f['name'] in source_text:
                        file_id = f['id']
                        break

            # Si tras el chequeo sigue sin encontrarse, asignamos 'desconocido'
            file_id = file_id or "desconocido"

            # Inyectar explícitamente en los metadatos del chunk
            chunk.metadata["file_id"] = file_id
            chunk.metadata["id"] = file_id  # Guardado redundante para retrocompatibilidad
            chunk.metadata["modifiedTime"] = dates_map.get(file_id, "desconocido")
            
            # Construir ID único determinista para el vector en Pinecone
            custom_id = f"{file_id}_chunk_{i}"
            ids.append(custom_id)

        print(f"Generando embeddings e indexando {len(chunks)} fragmentos en Pinecone...")

        # 5. Conectar con Pinecone e insertar los fragmentos con la metadata corregida
        vector_store = PineconeVectorStore(
            index_name = self.index_name,
            embedding = self.model
        )
        
        # Vincula cada chunk con el ID determinista de la lista 'ids'
        vector_store.add_documents(documents=chunks, ids=ids)

        print(f"✨ ¡Base de datos en Pinecone ('{self.index_name}') indexada exitosamente! ✨")

    def update_embeddings(self):
        """
          Sincronización incremental.
          Sincroniza Google Drive con Pinecone procesando únicamente
          los archivos nuevos o modificados.
        """
        print("🔍 Analizando carpeta de Google Drive...")
    
        # PASO 1: Obtener listado ligero de archivos en Drive
        google_drive_files = self._get_google_drive_metadata()
        if not google_drive_files:
            print("No se encontraron archivos PDF en la carpeta de Google Drive.")
            return

        files_to_process = []

        # PASOS 2 y 3: Comparar metadatos de Drive contra Pinecone
        print("🔄 Verificando cambios contra el índice de Pinecone...")
        
        DIMENSION_INDICE = 768  # Configurado a la dimensión de tu embeddings_model

        # 2. En la consulta de verificación dentro de actualizar_base_de_datos():
        # En lugar de un vector de ceros, podemos buscar directamente si existen IDs con ese prefijo
        # o hacer un fetch/query amplio sin depender de un filtro restrictivo:
        for file in google_drive_files:
            f_id = file['id']
            f_name = file['name']
            f_mod_time = file['modifiedTime']

            # Consultar intentando traer cualquier vector asociado a este file_id o id
            query_response = self.index.query(
                vector=[0.01] * DIMENSION_INDICE, # Valores pequeños distintos de cero
                top_k=5,
                include_metadata=True,
                filter={
                    "$or": [
                        {"file_id": {"$eq": f_id}},
                        {"id": {"$eq": f_id}}
                    ]
                }
            )

            matches = query_response.get("matches", [])

            print(f"\n🔍 Diagnóstico para: '{f_name}'")
            print(f"   📅 Fecha en Google Drive : '{f_mod_time}'")

            if matches:
                metadata_pinecone = matches[0].get("metadata", {})
                pinecone_saved_date = metadata_pinecone.get("modifiedTime")
                
                print(f"   📌 Fecha en Pinecone     : '{pinecone_saved_date}'")
                # print(f"   📦 Metadatos en Pinecone : {metadata_pinecone}")

                # Comparación en Python
                if pinecone_saved_date == f_mod_time:
                    print(f"   ✅ Sin cambios: '{f_name}' ya está actualizado.")
                else:
                    print(f"   ⚠️ Fechas diferentes ➔ Marcado para actualización.")
                    files_to_process.append(file)
            else:
                print(f"   🔴 No existe en Pinecone (0 matches por file_id) ➔ Archivo nuevo.")
                files_to_process.append(file)

        # PASO 5: Evaluar si hay trabajo por hacer
        if not files_to_process:
            print("\n✨ La base de datos ya está 100% sincronizada. No hay cambios que procesar.")
            return

        print(f"\n🚀 Procesando {len(files_to_process)} archivo(s)...")

        # Limpiar vectores previos de los archivos que van a re-indexarse (Paso 5 cont.)
        for file in files_to_process:
            self._remove_specific_vector(file['id'])

        # PASO 6: Descargar e indexar ÚNICAMENTE los archivos de la lista
        selected_file_ids = [f['id'] for f in files_to_process]
        
        loader = GoogleDriveLoader(
            credentials = self.google_credentials,
            file_ids = selected_file_ids,  # Carga únicamente los archivos filtrados
            recursive = False
        )
        
        documents = loader.load()

        # Inyectar metadata de modifiedTime y file_id de forma garantizada
        for doc in documents:
            # GoogleDriveLoader suele guardar el ID en 'id' o en 'source'
            doc_id = doc.metadata.get("id") or doc.metadata.get("file_id")
            
            # Si por alguna razón no está en metadata, intentamos mapear por el nombre del archivo
            doc_title = doc.metadata.get("title") or doc.metadata.get("source", "")

            for a in files_to_process:
                if a['id'] == doc_id or a['name'] in doc_title:
                    doc.metadata["modifiedTime"] = a['modifiedTime']
                    doc.metadata["file_id"] = a['id']
                    doc.metadata["id"] = a['id']  # Guardamos ambos para retrocompatibilidad
                    break

        # Dividir en fragmentos (chunks)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)

        # Construir IDs únicos y deterministas por chunk
        ids = []
        for i, chunk in enumerate(chunks):
            f_id = chunk.metadata.get("file_id", "desconocido")
            ids.append(f"{f_id}_chunk_{i}")

        # Cargar los nuevos embeddings a Pinecone
        vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.model
        )
        vector_store.add_documents(documents=chunks, ids=ids)

        print(f"\n✨ ¡Sincronización completada exitosamente en Pinecone ('{INDEX_NAME}')! ✨")


    # MÉTODOS AUXILIARES
    def _has_index_vectors(self) -> bool:
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0) > 0

    def _get_google_drive_metadata(self):
        service = build('drive', 'v3', credentials = self.google_credentials)
        query = f"'{self.folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name, modifiedTime)").execute()
        return results.get('files', [])

    def _remove_specific_vector(self, file_id: str):
        try:
            self.index.delete(filter={"file_id": {"$eq": file_id}})
            self.index.delete(filter={"id": {"$eq": file_id}})
        except Exception as e:
            print(f"⚠️ Nota al eliminar vectores para {file_id}: {e}")