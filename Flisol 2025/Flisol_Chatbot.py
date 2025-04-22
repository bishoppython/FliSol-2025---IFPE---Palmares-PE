import streamlit as st  # importa a biblioteca Streamlit para criar a interface web

import time             # importa o módulo time (não utilizado diretamente, mas mantido caso queira adicionar delays)

from langchain_community.document_loaders import PyPDFLoader # importa o carregador de PDFs da LangChain Community
                         
from langchain.text_splitter import RecursiveCharacterTextSplitter  # importa a classe para dividir o texto em pedaços menores
                         
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # importa o gerador de embeddings da API Generative AI do Google
                         
from langchain_chroma import Chroma   # importa o armazenamento vetorial Chroma para indexação e busca
                                   
from langchain_google_genai import ChatGoogleGenerativeAI # importa o wrapper da LangChain para chat usando o modelo Generative AI do Google
                         
from langchain.chains import create_retrieval_chain  # importa função para criar uma cadeia de recuperação (RAG)
                         
from langchain.chains.combine_documents import create_stuff_documents_chain  # importa função para combinar documentos recuperados em um único prompt
                         
from langchain_core.prompts import ChatPromptTemplate  # importa a classe para criar templates de prompt para chat
                         
from dotenv import load_dotenv  # importa função para carregar variáveis de ambiente de um arquivo .env

import glob

import json

import warnings
warnings.filterwarnings("ignore")  # ignora avisos de depreciação e outros avisos

# ─── CONFIGURAÇÃO STREAMLIT ──────────────────────────────────────────────────────

# ⬇️ CONFIGURAÇÃO DA PÁGINA (TÍTULO E ÍCONE DA ABA) ⬇️
st.set_page_config(
    page_title="Sobre a Banda PoliRockers",    # título que aparece na aba do navegador
    page_icon="📚"                         # emoji como favicon; pode ser caminho "favicon.png"
    # layout="wide"                          # opcional, deixa a página em modo “wide”
)

load_dotenv()                    # carrega as variáveis definidas em .env para o ambiente


# ─── CARREGAMENTO DE DOCUMENTOS ──────────────────────────────────────────────────


# Define o título da aplicação que será exibido no topo da interface Streamlit
st.title("Banda PoliRockers")

# 1) Encontre todos os PDFs em docs/
# pdf_paths = glob.glob("/mnt/A06815BA68159060/FLISOL/2025/docs/*.pdf")

################### Para utilizar apenas com um Pdf Especifico ########################
# Carrega o PDF especificado a partir do caminho local
loader = PyPDFLoader("/mnt/A06815BA68159060/FLISOL/2025/docs/polirockers.pdf")
data = loader.load()  # realiza a leitura do PDF e retorna uma lista de objetos de página

# 2) Carregue e agregue todas as páginas de todos os PDFs
# all_pages = []
# for path in pdf_paths:
#     loader = PyPDFLoader(path)
#     pages = loader.load()        # retorna lista de páginas do PDF atual
#     all_pages.extend(pages)      # adiciona ao conjunto geral


# Configura o splitter para dividir o texto em pedaços de até 1000 caracteres, mantendo contexto
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data) # Caso utilize apenas um pdf
# docs = text_splitter.split_documents(all_pages)
# docs agora é uma lista de "documentos" menores, prontos para indexação


# Cria o armazenamento vetorial (vector store) a partir dos documentos fragmentados
# Utiliza embeddings gerados pelo modelo "embedding-001" da Google Generative AI
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    # persist_directory="./chroma_db"    # ← cria ./chroma_db e o tenant default_tenant automaticamente
)

# Cria um retriever que buscará os trechos mais similares no vectorstore
# Configura para retornar os top 10 resultados por similaridade
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})


# Inicializa o LLM de chat, apontando para o modelo "gemini-1.5-flash-latest"
# Define temperatura zero para respostas determinísticas e sem limite de tokens
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0,
    max_tokens=None,
    timeout=None
)


# Cria um campo de input de chat na interface; o conteúdo digitado pelo usuário vai para "query" - BKP
query = st.chat_input("Faz teu nome: ")
prompt = query  # variável temporária que receberá o prompt do usuário


# Define o prompt de sistema (system message) que orienta o comportamento do assistente
system_prompt = (
    "Você é um assistente de perguntas e respostas. Utilize os trechos de contexto recuperado abaixo para responder à pergunta. Se não souber a resposta, diga que não sabe, ignore todo e qualquer questão fora do contexto dos dados. Use, no máximo, três frases e mantenha a resposta concisa."
    "\n\n"
    "{context}"  # placeholder que será preenchido com os trechos recuperados
)


# Constrói um template de prompt para chat, mesclando a system message e a mensagem humana
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # define instruções de sistema
        ("human", "{input}"),       # define onde entra a pergunta do usuário
    ]
)

# ─── SESSÃO PARA HISTÓRICO ────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # cada item será dict: {"user": ..., "bot": ...}
    
# ─── INPUT DE USUÁRIO E RESPOSTA ─────────────────────────────────────────────────
# query = st.chat_input("Faça sua pergunta:")
# if query:
#     qa_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever, qa_chain)
#     result = rag_chain.invoke({"input": query})
#     answer = result["answer"]
#     # salva no histórico
#     st.session_state.history.append({"user": query, "bot": answer})

# Quando o usuário enviar algo no chat... # BKP
if query:
    # Cria uma subcadeia que combina documentos recuperados em um único prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # Cria a cadeia completa de RAG: busca no retriever + passa para a subcadeia Q&A
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoca a cadeia com a entrada do usuário
    response = rag_chain.invoke({"input": query})
    # response["answer"] contém a resposta gerada pelo modelo

    # Exibe a resposta gerada na interface Streamlit
    st.write(response["answer"])

# ─── EXIBE HISTÓRICO NA TELA ─────────────────────────────────────────────────────
# st.markdown("#### Histórico de Conversa")
# for entry in st.session_state.history:
#     st.markdown(f"**Você:** {entry['user']}")
#     st.markdown(f"**Assistente:** {entry['bot']}")
#     st.write("---")

# ─── DOWNLOAD DO HISTÓRICO ───────────────────────────────────────────────────────
# if st.session_state.history:
#     history_json = json.dumps(st.session_state.history, ensure_ascii=False, indent=2)
#     st.download_button(
#         label="📥 Baixar Histórico",
#         data=history_json,
#         file_name="chat_history.json",
#         mime="application/json"
#     )