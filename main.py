from config import embeddings_model, llm, INDEX_NAME   
from embeddings import PineconeService
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

def format_documents(docs):
    formatted_chunks = []
    
    for doc in docs:
        nombre_doc = doc.metadata.get("file_name") or doc.metadata.get("source", "Documento Senkats")
        estatus = doc.metadata.get("doc_status", "No especificado")
        ultima_actualizacion = doc.metadata.get("doc_last_update", "No especificada")
        
        chunk_text = (
            f"--- FUENTE DE INFORMACIÓN ---\n"
            f"Documento: {nombre_doc}\n"
            f"Estatus de vigencia: {estatus}\n"
            f"Última fecha de actualización: {ultima_actualizacion}\n"
            f"Contenido:\n{doc.page_content}\n"
            f"--- FIN FUENTE ---"
        )
        formatted_chunks.append(chunk_text)
        
    return "\n\n".join(formatted_chunks)

def preguntar_al_agente(pregunta: str) -> str:
    """Consulta la base de datos vectorial remota de Pinecone y responde utilizando el LLM, Gemini."""
    
    # 1. Conectarse al índice existente en Pinecone
    vectorstore = PineconeVectorStore(
        index_name = INDEX_NAME,
        embedding = embeddings_model
    )
    
    # 2. Configurar el recuperador (top 4 fragmentos)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 3. Definir la plantilla del prompt del sistema
    system_prompt = (
        """
            Tu nombre es Zorawaru 🦊, un asistente amigable, divertido y profesional experto en la plataforma educativa 'Senkats'. 

            ### OBJETIVO
            Responder las dudas de los usuarios utilizando ÚNICAMENTE la información del contexto proporcionado.

            ### REGLAS DE RESPUESTA
            1. Mantén un tono amable, jovial y profesional. Puedes usar emojis adecuados al contexto 🚀.
            2. Básate estrictamente en el contexto. Si la respuesta no está en el contexto o no estás seguro, indica abiertamente que no encuentras la información. No inventes datos.
            3. Lee atentamente la cabecera o metadata de los fragmentos en el contexto para identificar los datos del documento origen.
            4. DEDUPLICACIÓN Y AGRUPACIÓN DE FUENTES:
                - Si la información proviene del documento en general o abarca toda una categoría/capítulo, indica el nombre del Capítulo o Categoría.
                - Si abarca múltiples subsecciones dentro del mismo fragmento (ej. 3.1 y 3.2), lista los apartados correspondientes (ej. "Apartado: 3.1 y 3.2" o "Apartado: Completo / Varios").
                - Solo usa "No especificado en el fragmento" si de verdad el fragmento no contiene ningún encabezado, número de artículo o subsección visible.

            ### ESTRUCTURA DE SALIDA
            Responde a la pregunta del usuario y finaliza OBLIGATORIAMENTE con el bloque de "Fuentes consultadas". 

            [Tu respuesta detallada y clara aquí]

            ---
            **Fuentes consultadas:**
            Fuente #<índice>
            - **Categoría/Sección:** <Categoría, capítulo o sección correspondiente> | **Apartado:** <Subsección(es), número(s) de apartado (ej. 3.1 y 3.2), o "Sección completa">
            - **Documento:** <Nombre oficial del documento o código>
            - **Estatus:** <Vigente / Inactivo / No especificado en el fragmento>
            - **Última actualización:** <Fecha de última actualización o No especificada en el fragmento>
            \n\n

            (Nota: Si la respuesta proviene de múltiples fragmentos o archivos, enumera cada fuente en el formato anterior).

            ### CONTEXTO PROPORCIONADO:
            {context}
        """
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{question}"),
    ])

    # 4. Cadena RAG moderna usando LCEL
    rag_chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Ejecutar la consulta (retorna el string de la respuesta directamente)
    respuesta = rag_chain.invoke(pregunta)
    return respuesta


if __name__ == "__main__":
    # PASO A: Cargar, o revisar si es necesario actualizar los vectores en la nube de Pinecone
    sync_service = PineconeService()
    sync_service.sync()

    # PASO B: Preguntar al agente
    # respuesta = preguntar_al_agente("¿Qué es la Pawademia?")
    # respuesta = preguntar_al_agente("¿La plataforma tiene un curso sobre Inteligencia Artificial?")
    # respuesta = preguntar_al_agente("¿Quién es el fundador de la plataforma?")

    respuesta = preguntar_al_agente("¿Cómo puedo hablar al servicio al cliente?")

    print(f"\n🤖Respuesta del agente:\n{respuesta}")
