import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List
import streamlit as st
import tiktoken
from streamlit_chat import message
import os
from dotenv import load_dotenv
load_dotenv()
NOTION_API_KEY = os.getenv('NOTION_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

BASE_URL = "https://api.notion.com"

def notion_get_blocks(page_id: str, headers: dict):
    res = requests.get(f"{BASE_URL}/v1/blocks/{page_id}/children?page_size=100", headers=headers)
    return res.json()

def notion_search(query: dict, headers: dict):
    res = requests.post(f"{BASE_URL}/v1/search", headers=headers, data=query)
    return res.json()

def get_page_text(page_id: str, headers: dict):
    page_text = []
    try:
        blocks = notion_get_blocks(page_id, headers)
        for item in blocks.get('results', []):
            item_type = item.get('type')
            content = item.get(item_type, {})

            # 텍스트 블록 처리
            rich_texts = content.get('rich_text', [])
            if rich_texts and isinstance(rich_texts, list):
                for text in rich_texts:
                    plain_text = text.get('plain_text', '')
                    if plain_text:
                        page_text.append(plain_text)

            # 코드 블록 처리
            elif item_type == 'code':
                rich_texts = content.get('rich_text', [])
                if rich_texts and isinstance(rich_texts, list):
                    code = rich_texts[0].get('plain_text', '')
                    if code:
                        page_text.append(f"코드: {code}")

            # 이미지 블록 처리
            elif item_type == 'image':
                image_url = content.get('file', {}).get('url', '')
                if image_url:
                    page_text.append(f"이미지: {image_url}")

            # 제목 블록 처리
            elif item_type in ['heading_1', 'heading_2', 'heading_3']:
                rich_texts = content.get('rich_text', [])
                if rich_texts and isinstance(rich_texts, list):
                    heading = rich_texts[0].get('plain_text', '')
                    if heading:
                        page_text.append(f"제목: {heading}")

            # 리스트 블록 처리
            elif item_type in ['bulleted_list_item', 'numbered_list_item']:
                rich_texts = content.get('rich_text', [])
                if rich_texts and isinstance(rich_texts, list):
                    list_item = rich_texts[0].get('plain_text', '')
                    if list_item:
                        page_text.append(f"• {list_item}")

        return page_text
    except Exception as e:
        st.error(f"페이지 내용을 가져오는 중 오류가 발생했습니다: {str(e)}")
        return []

def load_notion(headers: dict) -> list:
    documents = []
    try:
        all_notion_documents = notion_search({}, headers)
        st.write("API 응답:", all_notion_documents)
        
        items = all_notion_documents.get('results', [])
        if not items:
            st.error("Notion에서 문서를 찾을 수 없습니다. API 키와 권한을 확인해주세요.")
            return documents

        for item in items:
            object_type = item.get('object')
            object_id = item.get('id')
            url = item.get('url')
            title = ""
            page_text = []

            if object_type == 'page':
                title_content = item.get('properties', {}).get('title')
                if title_content:
                    title = title_content.get('title', [{}])[0].get('text', {}).get('content', '')
                elif item.get('properties', {}).get('Name'):
                    if len(item.get('properties', {}).get('Name', {}).get('title', [])) > 0:
                        title = item.get('properties', {}).get('Name', {}).get('title', [{}])[0].get('text', {}).get('content', '')

                page_text.append([title])
                page_content = get_page_text(object_id, headers)
                page_text.append(page_content)

                flat_list = [item for sublist in page_text for item in sublist]
                text_per_page = ". ".join(flat_list)
                if len(text_per_page) > 0:
                    documents.append(text_per_page)

        if not documents:
            st.warning("Notion에서 가져온 문서에 내용이 없습니다.")
            
    except Exception as e:
        st.error(f"Notion API 호출 중 오류가 발생했습니다: {str(e)}")
    
    return documents

def chunk_tokens(text: str, token_limit: int) -> list:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    chunks = []
    tokens = tokenizer.encode(text, disallowed_special=())

    while tokens:
        chunk = tokens[:token_limit]
        chunk_text = tokenizer.decode(chunk)
        last_punctuation = max(
            chunk_text.rfind("."),
            chunk_text.rfind("?"),
            chunk_text.rfind("!"),
            chunk_text.rfind("\n"),
        )
        if last_punctuation != -1:
            chunk_text = chunk_text[: last_punctuation + 1]
        cleaned_text = chunk_text.replace("\n", " ").strip()

        if cleaned_text and (not cleaned_text.isspace()):
            chunks.append(cleaned_text)
        tokens = tokens[len(tokenizer.encode(chunk_text, disallowed_special=())):]

    return chunks

@st.cache_resource
def connect_to_vectorstore():
    client = QdrantClient(host="qdrant", port=6333)
    try:
        client.get_collection("notion_streamlit")
    except Exception as e:
        client.recreate_collection(
            collection_name="notion_streamlit",
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        )
    return client

def load_data_into_vectorstore(qdrant_client, docs: List[str], openai_api_key: str):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Qdrant(client=qdrant_client, collection_name="notion_streamlit", embedding_function=embeddings.embed_query)
    ids = vectorstore.add_texts(docs)
    return ids

@st.cache_data
def cache_headers(notion_api_key: str):
    headers = {"Authorization": f"Bearer {notion_api_key}", "Content-Type": "application/json",
               "Notion-Version": "2022-06-28"}
    return headers

@st.cache_resource
def load_chain(_client, api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Qdrant(client=_client, collection_name="notion_streamlit", embedding_function=embeddings.embed_query)
    chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo',
            openai_api_key=api_key),
            retriever=vectorstore.as_retriever()
    )
    return chain

def main():
    st.title('Notion 문서와 대화하기!')

    # API 키 확인
    if not OPENAI_API_KEY or not NOTION_API_KEY:
        st.error("환경 변수에 API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return

    vector_store = connect_to_vectorstore()
    
    with st.sidebar:
        notion_headers = cache_headers(NOTION_API_KEY)

        load_data = st.button('데이터 로드')
        if load_data:
            with st.spinner('Notion 문서를 로드하는 중...'):
                documents = load_notion(notion_headers)

                if documents:
                    chunks = []
                    for doc in documents:
                        chunks.extend(chunk_tokens(doc, 100))

                    with st.spinner('벡터 저장소에 데이터를 저장하는 중...'):
                        load_data_into_vectorstore(vector_store, chunks, OPENAI_API_KEY)
                        st.success("문서가 로드되었습니다.")
                else:
                    st.error("로드할 문서가 없습니다.")

    chain = load_chain(vector_store, OPENAI_API_KEY)

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    user_input = st.text_input("사용자: ", placeholder="Notion 문서와 대화해보세요 👇", key="input")

    if user_input:
        result = chain({"question": user_input, "chat_history": st.session_state["generated"]})
        response = result['answer']

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append((user_input, result["answer"]))

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i][1], key=str(i))

if __name__ == "__main__":
    main()