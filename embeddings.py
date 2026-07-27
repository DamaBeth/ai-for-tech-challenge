import re
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
        """Coordinador principal"""
        if not self._has_index_vectors():
            print("🚀 Primera vez: Cargando e indexando documentos desde Google Drive...")
            self.create_embeddings()
        else:
            print("🔄 Sincronizando cambios con Google Drive y Pinecone...")
            self.update_embeddings()

    def create_embeddings(self):
        """Indexación desde cero filtrando documentos vigentes."""
        service = build('drive', 'v3', credentials=self.google_credentials)
        results = service.files().list(
            q=f"'{self.folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            fields="files(id, name, modifiedTime)"
        ).execute()
        
        files_list = results.get('files', [])
        names_map = {f['id']: f['name'] for f in files_list}
        dates_map = {f['id']: f['modifiedTime'] for f in files_list}

        loader = GoogleDriveLoader(
            folder_id=self.folder_id,
            credentials=self.google_credentials,
            recursive=False,
            file_types=["application/pdf"]
        )
        
        documents = loader.load()
        print(f"Se encontraron {len(documents)} archivos PDF en Google Drive.")

        if not documents:
            return

        # 1. Unificar texto por fuente/título
        doc_header_metadata = {}
        docs_by_source = {}
        for doc in documents:
            source_key = doc.metadata.get("source") or doc.metadata.get("title") or doc.metadata.get("id")
            if source_key not in docs_by_source:
                docs_by_source[source_key] = ""
            docs_by_source[source_key] += "\n" + doc.page_content

        # 2. Extraer metadata y verificar vigencia
        valid_doc_ids = set()
        for doc in documents:
            f_id = doc.metadata.get("id") or doc.metadata.get("file_id")
            source_key = doc.metadata.get("source") or doc.metadata.get("title") or f_id
            
            if not f_id or f_id == "desconocido":
                for f in files_list:
                    if f['id'] in str(source_key) or f['name'] in str(source_key):
                        f_id = f['id']
                        break

            if f_id and f_id != "desconocido" and f_id not in doc_header_metadata:
                full_text = docs_by_source.get(source_key, doc.page_content)
                extracted = self._extract_header_metadata(full_text)
                file_name = names_map.get(f_id, "Sin nombre")

                # REGLA: Filtrar documentos NO vigentes
                if not self._is_status_valid(extracted['status']):
                    print(f"🚫 Ignorando documento no vigente: [{file_name}] (Estatus: {extracted['status']})")
                    continue

                doc_header_metadata[f_id] = extracted
                valid_doc_ids.add(f_id)
                print(f"📄 Archivo VIGENTE [{file_name}] ➔ Estatus: {extracted['status']} | Actualización: {extracted['last_update']}")

        # Filtrar solo los documentos que pertenecen a IDs vigentes
        documents_to_index = [
            doc for doc in documents 
            if (doc.metadata.get("id") or doc.metadata.get("file_id")) in valid_doc_ids or
               any(f_id in str(doc.metadata.get("source", "")) for f_id in valid_doc_ids)
        ]

        if not documents_to_index:
            print("⚠️ No hay documentos vigentes para indexar.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents_to_index)
        print(f"Generando embeddings para {len(chunks)} fragmentos...")

        ids = []
        for i, chunk in enumerate(chunks):
            file_id = chunk.metadata.get("id") or chunk.metadata.get("file_id")

            if not file_id or file_id == "desconocido":
                source_text = chunk.metadata.get("source", "") or chunk.metadata.get("title", "")
                for f in files_list:
                    if f['id'] in source_text or f['name'] in source_text:
                        file_id = f['id']
                        break

            file_id = file_id or "desconocido"
            file_name = names_map.get(file_id, chunk.metadata.get("title", "Documento Senkats"))
            drive_mod_time = dates_map.get(file_id, "desconocido")
            
            header_meta = doc_header_metadata.get(file_id, {
                "last_update": "No especificada en el documento",
                "status": "Vigente"
            })

            # Inyección en Metadatos
            chunk.metadata["file_id"] = file_id
            chunk.metadata["id"] = file_id
            chunk.metadata["file_name"] = file_name
            chunk.metadata["source"] = file_name
            chunk.metadata["modifiedTime"] = drive_mod_time
            chunk.metadata["doc_last_update"] = header_meta["last_update"]
            chunk.metadata["doc_status"] = header_meta["status"]

            header_prefix = (
                f"[DOCUMENTO: {file_name} | "
                f"ESTATUS: {header_meta['status']} | "
                f"ÚLTIMA ACTUALIZACIÓN OFICIAL: {header_meta['last_update']}]\n\n"
            )
            if not chunk.page_content.startswith("[DOCUMENTO:"):
                chunk.page_content = header_prefix + chunk.page_content

            ids.append(f"{file_id}_chunk_{i}")

        vector_store = PineconeVectorStore(index_name=self.index_name, embedding=self.model)
        vector_store.add_documents(documents=chunks, ids=ids)
        print(f"✨ ¡Base de datos indexada exitosamente con {len(chunks)} fragmentos! ✨")

    def update_embeddings(self):
        """Sincronización incremental: elimina borrados, ignora no vigentes y actualiza modificados."""
        google_drive_files = self._get_google_drive_metadata()
        drive_file_ids = {f['id'] for f in google_drive_files}

        # -------------------------------------------------------------
        # PASO 1: Eliminar archivos huérfanos (Borrados de Google Drive)
        # -------------------------------------------------------------
        pinecone_file_ids = self._get_all_pinecone_file_ids()
        orphan_ids = pinecone_file_ids - drive_file_ids

        if orphan_ids:
            print(f"🗑️ Se detectaron {len(orphan_ids)} archivos eliminados de Google Drive. Removiendo de Pinecone...")
            for orphan_id in orphan_ids:
                self._remove_specific_vector(orphan_id)
                print(f"   ❌ Vectores eliminados para ID: {orphan_id}")

        if not google_drive_files:
            print("✨ No hay archivos en Google Drive.")
            return

        # -------------------------------------------------------------
        # PASO 2: Identificar archivos nuevos o modificados
        # -------------------------------------------------------------
        files_to_process = []
        DIMENSION_INDICE = 768

        for file in google_drive_files:
            f_id = file['id']
            f_mod_time = file['modifiedTime']

            query_response = self.index.query(
                vector=[0.01] * DIMENSION_INDICE,
                top_k=1,
                include_metadata=True,
                filter={"$or": [{"file_id": {"$eq": f_id}}, {"id": {"$eq": f_id}}]}
            )

            matches = query_response.get("matches", [])
            if matches:
                metadata_pinecone = matches[0].get("metadata", {})
                pinecone_saved_date = metadata_pinecone.get("modifiedTime")

                if pinecone_saved_date != f_mod_time:
                    files_to_process.append(file)
            else:
                files_to_process.append(file)

        if not files_to_process and not orphan_ids:
            print("\n✨ La base de datos vectorial ya está 100% actualizada.")
            return

        if not files_to_process:
            return

        # -------------------------------------------------------------
        # PASO 3: Procesar archivos nuevos/modificados
        # -------------------------------------------------------------
        selected_file_ids = [f['id'] for f in files_to_process]
        loader = GoogleDriveLoader(
            credentials=self.google_credentials,
            file_ids=selected_file_ids,
            recursive=False
        )
        documents = loader.load()

        doc_header_metadata = {}
        docs_by_source = {}
        for doc in documents:
            source_key = doc.metadata.get("source") or doc.metadata.get("title") or doc.metadata.get("id")
            if source_key not in docs_by_source:
                docs_by_source[source_key] = ""
            docs_by_source[source_key] += "\n" + doc.page_content

        valid_doc_ids = set()
        for doc in documents:
            f_id = doc.metadata.get("id") or doc.metadata.get("file_id")
            source_key = doc.metadata.get("source") or doc.metadata.get("title") or f_id
            
            if not f_id or f_id == "desconocido":
                for f in files_to_process:
                    if f['id'] in str(source_key) or f['name'] in str(source_key):
                        f_id = f['id']
                        break

            if f_id and f_id != "desconocido" and f_id not in doc_header_metadata:
                full_text = docs_by_source.get(source_key, doc.page_content)
                extracted = self._extract_header_metadata(full_text)
                
                file_name = next((f['name'] for f in files_to_process if f['id'] == f_id), "Sin nombre")

                # Limpiamos versión previa en Pinecone de todas formas
                self._remove_specific_vector(f_id)

                # REGLA: Si NO está vigente, no lo indexamos
                if not self._is_status_valid(extracted['status']):
                    print(f"🚫 Se omitió e ignoró por no estar vigente: [{file_name}] (Estatus: {extracted['status']})")
                    continue

                doc_header_metadata[f_id] = extracted
                valid_doc_ids.add(f_id)
                print(f"📄 Procesando archivo VIGENTE [{file_name}] ➔ Estatus: {extracted['status']}")

        # Filtrar documentos a procesar
        documents_to_index = [
            doc for doc in documents 
            if (doc.metadata.get("id") or doc.metadata.get("file_id")) in valid_doc_ids or
               any(f_id in str(doc.metadata.get("source", "")) for f_id in valid_doc_ids)
        ]

        if not documents_to_index:
            print("✨ Sincronización completada. No hubo nuevos documentos vigentes para indexar.")
            return

        files_dict = {f['id']: f for f in files_to_process}
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents_to_index)

        ids = []
        for i, chunk in enumerate(chunks):
            doc_id = chunk.metadata.get("id") or chunk.metadata.get("file_id")
            doc_title = chunk.metadata.get("title") or chunk.metadata.get("source", "")

            matched_file = files_dict.get(doc_id)
            if not matched_file:
                for f in files_to_process:
                    if f['name'] in doc_title:
                        matched_file = f
                        break

            file_id = matched_file['id'] if matched_file else (doc_id or "desconocido")
            file_name = matched_file['name'] if matched_file else doc_title
            drive_mod_time = matched_file['modifiedTime'] if matched_file else "desconocido"

            header_meta = doc_header_metadata.get(file_id, {
                "last_update": "No especificada en el documento",
                "status": "Vigente"
            })

            chunk.metadata["file_id"] = file_id
            chunk.metadata["id"] = file_id
            chunk.metadata["file_name"] = file_name
            chunk.metadata["source"] = file_name
            chunk.metadata["modifiedTime"] = drive_mod_time
            chunk.metadata["doc_last_update"] = header_meta["last_update"]
            chunk.metadata["doc_status"] = header_meta["status"]

            header_prefix = (
                f"[DOCUMENTO: {file_name} | "
                f"ESTATUS: {header_meta['status']} | "
                f"ÚLTIMA ACTUALIZACIÓN OFICIAL: {header_meta['last_update']}]\n\n"
            )
            if not chunk.page_content.startswith("[DOCUMENTO:"):
                chunk.page_content = header_prefix + chunk.page_content

            ids.append(f"{file_id}_chunk_{i}")

        vector_store = PineconeVectorStore(index_name=self.index_name, embedding=self.model)
        vector_store.add_documents(documents=chunks, ids=ids)
        print(f"\n✨ Sincronización completada exitosamente en Pinecone. ✨")

    # -------------------------------------------------------------
    # MÉTODOS AUXILIARES
    # -------------------------------------------------------------
    def _is_status_valid(self, status: str) -> bool:
        """Verifica si el estatus extraído del documento se considera vigente."""
        if not status:
            return True
        status_clean = status.lower().strip()
        invalid_keywords = ["inactivo", "obsoleto", "derogado", "no vigente", "expirado", "archivado"]
        return not any(kw in status_clean for kw in invalid_keywords)

    def _get_all_pinecone_file_ids(self) -> set:
        """Consulta el índice para obtener el conjunto de file_ids únicos presentes en Pinecone."""
        try:
            DIMENSION_INDICE = 768
            response = self.index.query(
                vector=[0.01] * DIMENSION_INDICE,
                top_k=10000,
                include_metadata=True
            )
            file_ids = set()
            for match in response.get("matches", []):
                meta = match.get("metadata", {})
                f_id = meta.get("file_id") or meta.get("id")
                if f_id:
                    file_ids.add(f_id)
            return file_ids
        except Exception as e:
            print(f"⚠️ Error al consultar IDs en Pinecone: {e}")
            return set()

    def _has_index_vectors(self) -> bool:
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0) > 0

    def _get_google_drive_metadata(self):
        service = build('drive', 'v3', credentials=self.google_credentials)
        query = f"'{self.folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        results = service.files().list(q=query, fields="files(id, name, modifiedTime)").execute()
        return results.get('files', [])

    def _remove_specific_vector(self, file_id: str):
        try:
            self.index.delete(filter={"file_id": {"$eq": file_id}})
            self.index.delete(filter={"id": {"$eq": file_id}})
        except Exception as e:
            print(f"⚠️ Nota al eliminar vectores para {file_id}: {e}")

    def _extract_header_metadata(self, text: str) -> dict:
        metadata = {
            "last_update": "No especificada en el documento",
            "status": "Vigente"
        }
        
        update_match = re.search(
            r"Última\s+Actualización\s*:\s*(.*?)(?=\s*(?:Estatus|Presencia|Versión|\n\n|$))", 
            text, 
            re.IGNORECASE | re.DOTALL
        )
        if update_match:
            val = update_match.group(1).strip()
            val = " ".join(val.split())
            if val:
                metadata["last_update"] = val

        status_match = re.search(
            r"Estatus\s*:\s*([^\n\r]+)", 
            text, 
            re.IGNORECASE
        )
        if status_match:
            val = status_match.group(1).strip()
            if val:
                metadata["status"] = val

        return metadata