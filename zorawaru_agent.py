from config import embeddings_model, llm, INDEX_NAME   
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore

def format_documents(docs):
    formatted_chunks = []
    
    for doc in docs:
        document_name = doc.metadata.get("file_name") or doc.metadata.get("source", "Documento Senkats")
        status = doc.metadata.get("doc_status", "No especificado")
        last_updated = doc.metadata.get("doc_last_update", "No especificada")
        
        chunk_text = (
            f"--- FUENTE DE INFORMACIÓN ---\n"
            f"Documento: {document_name}\n"
            f"Estatus de vigencia: {status}\n"
            f"Última fecha de actualización: {last_updated}\n"
            f"Contenido:\n{doc.page_content}\n"
            f"--- FIN FUENTE ---"
        )
        formatted_chunks.append(chunk_text)
        
    return "\n\n".join(formatted_chunks)


def ask_agent(pregunta: str, chat_history: list) -> str:
    """Consulta la base de datos vectorial y responde manteniendo el historial."""
    
    # 1. Conectarse al índice en Pinecone
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings_model
    )
    
    # 2. Configurar el recuperador
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 3. Recuperar y formatear los documentos relevantes para la pregunta actual
    docs = retriever.invoke(pregunta)
    contexto_formateado = format_documents(docs)

    # 4. Definir la plantilla del prompt del sistema con soporte para historial
    system_prompt = f"""
    Tu nombre es Zorawaru 🦊, un asistente amigable, divertido y profesional experto en la plataforma educativa 'Senkats'. 

    ### OBJETIVO
    Responder las dudas de los usuarios utilizando ÚNICAMENTE la información del contexto proporcionado y recordando la conversación previa.

    ### REGLAS DE RESPUESTA
    1. Mantén un tono amable, jovial y profesional. Puedes usar emojis adecuados al contexto 🚀.
    2. Saluda de manera eufórica ÚNICAMENTE en el primer mensaje de la conversación. En los mensajes posteriores, responde directamente a la pregunta sin volver a saludar de forma repetitiva.
    3. Básate estrictamente en el contexto. Si la respuesta no está en el contexto o no estás seguro, indica abiertamente que no encuentras la información. No inventes datos.
    4. Lee atentamente la cabecera o metadata de los fragmentos en el contexto para identificar los datos del documento origen.
    5. DEDUPLICACIÓN Y AGRUPACIÓN DE FUENTES:
       - Si la información proviene del documento en general o abarca toda una categoría/capítulo, indica el nombre del Capítulo o Categoría.
       - Si abarca múltiples subsecciones dentro del mismo fragmento (ej. 3.1 y 3.2), lista los apartados correspondientes (ej. "Apartado: 3.1 y 3.2" o "Apartado: Completo / Varios").
       - Solo usa "No especificado en el fragmento" si de verdad el fragmento no contiene ningún encabezado, número de artículo o subsección visible.

    ### ESTRUCTURA DE SALIDA
    Responde a la pregunta del usuario y finaliza OBLIGATORIAMENTE con el bloque de "Fuentes consultadas" en este HTML exacto:

    [Tu respuesta aquí]

    <details>
      <summary>📚 Fuentes consultadas (Clic para desplegar)</summary>
      <br>
      <b>Fuente #1</b>
      <ul>
        <li><b>Categoría/Sección:</b> ... | <b>Apartado:</b> ...</li>
        <li><b>Documento:</b> <code>...</code></li>
        <li><b>Estatus:</b> ...</li>
        <li><b>Última actualización:</b> ...</li>
      </ul>
    </details>


    ### CONTEXTO PROPORCIONADO:
    {contexto_formateado}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{question}"),
    ])

    # 5. Cadena RAG
    rag_chain = prompt | llm | StrOutputParser()

    # 6. Ejecutar pasando la pregunta y el historial
    respuesta = rag_chain.invoke({
        "question": pregunta,
        "chat_history": chat_history
    })

    return respuesta