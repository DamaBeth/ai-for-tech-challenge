from config import embeddings_model, llm, INDEX_NAME   
from embeddings import PineconeService
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

def format_documents(docs):
    """Auxiliar para unir el contenido de los fragmentos recuperados."""
    return "\n\n".join(doc.page_content for doc in docs)

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
            Tu nombre es 'Zorawaru' y eres un bot que asiste a los usuarios que tienen dudas sobre la plataforma educativa 'Senkats'.
            Responde a las preguntas utilizando únicamente el siguiente contexto extraído de los documentos de la empresa.
            Si no sabes la respuesta o no está en el contexto, di abiertamente que no la encuentras.
            Sé amable y divertido pero siempre profesional. Puedes utilizar emojis.
            Al responder cada pregunta menciona:
              - El nombre del documento del que obtuviste la información
              - El estatus del archivo
              - La última fecha de actualización
              - La categoría, capítulo, sección o artículo (según corresponda)
              - El número de apartado al que corresponde
            Ya que este será el formato es el estándar para cada referencia.
            Si obtuviste la respuesta de más de un archivo, siéntete libre de mencionar a cada uno en forma de lista siguiendo 
            el mismo estándar de respuesta por referencia.

            Ejemplo de pregunta del usuario: '¿Qué significa Senkats?'
            Ejemplo del formato de salida: 
                                            'Tu respuesta a la pregunta aquí.\n

                                            Referencias:
                                            Categoría: <Tu respuesta del nombre de la categoría>, Apartado <Número de apartado>
                                            Origen: <Tu respuesta del nombre del archivo>
                                            Estatus: <Tu respuesta del estatus del archivo>
                                            Última fecha de actualización: <Tu respuesta de la última fecha de actualización según el archivo>\n'
            
            El contexto es el siguiente:\n{context}
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
    respuesta = preguntar_al_agente("¿Qué es la Pawademia?")
    print(f"\n🤖Respuesta del agente:\n{respuesta}")

    # TODO: Retornar en la respuesta del agente la cita de las fuentes