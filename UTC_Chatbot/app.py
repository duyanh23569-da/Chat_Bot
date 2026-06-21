import ollama
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

VECTOR_FOLDER = "vectorstore/"

st.set_page_config(
    page_title="UTC Chatbot",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700&family=Playfair+Display:wght@700&display=swap');

/* ===== ROOT VARIABLES ===== */
:root {
    --bg-primary: #04080f;
    --bg-card: #080e1a;
    --bg-glass: rgba(255,255,255,0.03);
    --border: rgba(255,255,255,0.07);
    --border-glow: rgba(99, 179, 237, 0.3);
    --accent-blue: #3b9eff;
    --accent-gold: #f6c860;
    --accent-teal: #2dd4bf;
    --text-primary: #e8edf5;
    --text-muted: #5a6680;
    --text-caption: #8898aa;
    --user-bg: #0d1829;
    --bot-bg: #060c17;
    --shadow-blue: 0 0 40px rgba(59, 158, 255, 0.08);
}

/* ===== GLOBAL RESET ===== */
* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 40% at 20% 10%, rgba(59,158,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 80%, rgba(45,212,191,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 50% 30% at 60% 30%, rgba(246,200,96,0.03) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ===== HIDE STREAMLIT CHROME ===== */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}

/* ===== LAYOUT ===== */
.block-container {
    max-width: 860px !important;
    padding: 2.5rem 1.5rem 6rem !important;
    margin: 0 auto !important;
    position: relative;
    z-index: 1;
}

/* ===== HEADER ===== */
.utc-header {
    text-align: center;
    padding: 2.5rem 0 2rem;
    position: relative;
}

.utc-logo-ring {
    width: 72px;
    height: 72px;
    margin: 0 auto 1.2rem;
    border-radius: 50%;
    background: linear-gradient(135deg, #0d2240, #0d1f3a);
    border: 1px solid var(--border-glow);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 32px;
    box-shadow: 0 0 0 6px rgba(59,158,255,0.06), 0 0 40px rgba(59,158,255,0.15);
    animation: pulse-ring 4s ease-in-out infinite;
}

@keyframes pulse-ring {
    0%, 100% { box-shadow: 0 0 0 6px rgba(59,158,255,0.06), 0 0 40px rgba(59,158,255,0.12); }
    50% { box-shadow: 0 0 0 10px rgba(59,158,255,0.03), 0 0 60px rgba(59,158,255,0.22); }
}

.utc-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    background: linear-gradient(135deg, #e8edf5 0%, #3b9eff 50%, #f6c860 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}

.utc-subtitle {
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.8rem;
}

.utc-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-blue), transparent);
    margin: 0 auto;
    border-radius: 2px;
}

/* ===== WELCOME CHIPS ===== */
.welcome-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin: 1.6rem 0 0.5rem;
}

.chip {
    background: var(--bg-glass);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.76rem;
    color: var(--text-caption);
    font-weight: 500;
    transition: all 0.2s;
    cursor: default;
}

.chip:hover {
    border-color: rgba(59,158,255,0.35);
    color: var(--accent-blue);
    background: rgba(59,158,255,0.06);
}

/* ===== CHAT MESSAGES ===== */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.3rem 0 !important;
    margin: 0 !important;
    animation: fadeSlideIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Avatar styling */
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    border-radius: 50% !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
}

/* Message bubble wrapper */
[data-testid="stChatMessage"] .stMarkdown {
    padding: 0;
}

[data-testid="stChatMessage"][aria-label*="user"] > div:last-child,
[data-testid="stChatMessage"][aria-label*="User"] > div:last-child {
    background: linear-gradient(135deg, #0d1f3c, #0a1628) !important;
    border: 1px solid rgba(59,158,255,0.15) !important;
    border-radius: 2px 16px 16px 16px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.3) !important;
}

[data-testid="stChatMessage"][aria-label*="assistant"] > div:last-child,
[data-testid="stChatMessage"][aria-label*="Assistant"] > div:last-child {
    background: linear-gradient(135deg, #07111f, #060e1a) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px 2px 16px 16px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.4) !important;
}

/* Markdown inside messages */
[data-testid="stChatMessage"] p {
    color: var(--text-primary) !important;
    font-size: 0.93rem !important;
    line-height: 1.75 !important;
    margin: 0 0 0.5em !important;
    font-weight: 400 !important;
}

[data-testid="stChatMessage"] p:last-child { margin-bottom: 0 !important; }

[data-testid="stChatMessage"] strong {
    color: #c8d8f0 !important;
    font-weight: 600 !important;
}

[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] ol {
    color: var(--text-primary) !important;
    font-size: 0.91rem !important;
    line-height: 1.75 !important;
    padding-left: 1.3em !important;
}

[data-testid="stChatMessage"] li { margin-bottom: 0.25em !important; }

[data-testid="stChatMessage"] code {
    background: rgba(59,158,255,0.1) !important;
    color: #7ec8e3 !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    font-size: 0.84em !important;
}

/* ===== SOURCE CAPTION ===== */
[data-testid="stCaptionContainer"] p {
    font-size: 0.73rem !important;
    color: var(--text-muted) !important;
    margin-top: 6px !important;
    font-style: italic !important;
    padding-left: 2px !important;
}

/* ===== SPINNER ===== */
[data-testid="stSpinner"] > div {
    color: var(--accent-blue) !important;
}

/* ===== CHAT INPUT ===== */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important; right: 0 !important;
    padding: 1rem 1.5rem 1.4rem !important;
    background: linear-gradient(to top, #04080f 70%, transparent) !important;
    z-index: 999 !important;
}

[data-testid="stChatInput"] > div {
    max-width: 820px !important;
    margin: 0 auto !important;
}

[data-testid="stChatInput"] textarea {
    background: rgba(8,14,26,0.95) !important;
    border: 1px solid rgba(59,158,255,0.2) !important;
    border-radius: 14px !important;
    color: var(--text-primary) !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 14px 50px 14px 18px !important;
    backdrop-filter: blur(12px) !important;
    transition: border-color 0.2s !important;
    box-shadow: 0 0 0 3px rgba(59,158,255,0) !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(59,158,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(59,158,255,0.08), 0 0 30px rgba(59,158,255,0.1) !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
    font-style: italic !important;
}

/* Send button */
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #1a5ec0, #0d3d8f) !important;
    border: 1px solid rgba(59,158,255,0.3) !important;
    border-radius: 10px !important;
    color: white !important;
    transition: all 0.2s !important;
}

[data-testid="stChatInput"] button:hover {
    background: linear-gradient(135deg, #2570d8, #1550b0) !important;
    box-shadow: 0 0 16px rgba(59,158,255,0.3) !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(59,158,255,0.18); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(59,158,255,0.35); }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("""
<div class="utc-header">
    <div class="utc-logo-ring">🎓</div>
    <div class="utc-title">UTC Tuyển Sinh</div>
    <div class="utc-subtitle">Trường Đại học Giao thông Vận tải</div>
</div>
""", unsafe_allow_html=True)

# ===== LOAD VECTOR DB =====
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3"
    )
    return Chroma(
        persist_directory=VECTOR_FOLDER,
        embedding_function=embeddings
    )

def retrieve(question, vs):
    docs = vs.max_marginal_relevance_search(
        question,
        k=4,
        fetch_k=10
    )
    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata.get("source", "?") for d in docs]))
    return context, sources
#khi nhập câu hỏi thì câu hỏi sẽ được hàm trên chuyển hóa thành các vector embedding
#sau đó hệ thống truy vấn trong ChromaDB để tìm ra các đoạn văn bản liên quan gần nhất
#và các đoạn này sẽ chuyển hóa thành các đoạn text

def generate(question, context):
    prompt = f"""
Bạn là trợ lý tư vấn tuyển sinh của Trường Đại học Giao thông Vận tải.

QUY TẮC:
1. Chỉ trả lời dựa trên phần THÔNG TIN được cung cấp.
2. Không tự suy đoán hoặc bổ sung kiến thức ngoài dữ liệu.
3. Nếu THÔNG TIN không đủ để trả lời, hãy nói:
"Không tìm thấy thông tin phù hợp trong dữ liệu."
4. Với câu hỏi có số liệu (chỉ tiêu, học phí, tổ hợp, mã ngành), trả lời đúng theo dữ liệu.
5. Trả lời ngắn gọn, rõ ràng, tiếng Việt.

THÔNG TIN:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
"""
    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content":
                "Bạn là chatbot RAG, tuyệt đối chỉ dùng context được cung cấp."
            },
            {
                "role":"user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
#Hệ thống sử dụng mô hình ngôn ngữ lớn chạy cục bộ thông qua Ollama (ví dụ: LLaMA 3).
	#Đầu vào: context + câu hỏi
	#Đầu ra: câu trả lời bằng tiếng Việt
#Prompt được thiết kế để đảm bảo:
	#Chỉ sử dụng thông tin trong context
	#Tránh trả lời sai hoặc “bịa” thông tin
#ollama: ứng dụng cho phép chạy LLM trên chính máy cá nhân

vs = load_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

if question := st.chat_input("✦  Hỏi về tuyển sinh, ngành học, điểm chuẩn..."):
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Đang tìm kiếm thông tin..."):
            context, sources = retrieve(question, vs)
            answer = generate(question, context)

        st.markdown(answer)
        st.caption(f"📎 Nguồn tham khảo · {' · '.join(sources)}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })