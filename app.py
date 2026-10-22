import os
import json
import sqlite3

from flask import Flask, request, Response, render_template
from ollama import Client


app = Flask(__name__)

# 跟 crawl_data.py 使用同一個地端模型。
OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma4:e4b"

# 跟 make_index.py 使用同一個向量模型與索引檔。
EMBED_MODEL_NAME = "BAAI/bge-m3"
INDEX_PATH = "./vector.index"
DB_PATH = "./news.db"
HISTORY_FILE = "chat_history.json"

# 每次檢索回傳幾篇新聞。
TOP_K = 3

SYSTEM_PROMPT = (
    "你是健康諮詢小幫手。"
    "請用繁體中文，依照這次提供的新聞摘要回答，不要編造摘要裡沒有的醫療細節。"
    "如果這次沒有提供新聞摘要，就根據前面的對話，把使用者剛才問過的問題整理成重點。"
)

client = Client(
    host=OLLAMA_HOST,
    timeout=600
)

# 向量模型很大，第一次提問時才載入。
encoder = None
faiss_index = None


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)


def build_ollama_messages(history):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    for msg in history:
        role = msg["role"]
        if role == "model":
            role = "assistant"

        messages.append({
            "role": role,
            "content": msg["text"],
        })

    return messages


def load_search_tools():
    """載入向量模型與索引。沒有索引時，請先執行 make_index.py。"""
    global encoder, faiss_index

    if encoder is not None and faiss_index is not None:
        return encoder, faiss_index

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            "找不到 vector.index。請先在 cases/RAG 執行 python make_index.py。"
        )

    import faiss
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(EMBED_MODEL_NAME)
    faiss_index = faiss.read_index(INDEX_PATH)
    return encoder, faiss_index


def search_news(question):
    """用問題的向量去找最相近的文章摘要。"""
    model, index = load_search_tools()

    embedding = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    scores, ids = index.search(embedding, TOP_K)
    news_ids = [int(news_id) for news_id in ids[0] if int(news_id) >= 0]

    if not news_ids:
        return []

    conn = sqlite3.connect(DB_PATH)
    try:
        rows = []
        for news_id in news_ids:
            row = conn.execute(
                """
                SELECT title, link, date, created_at, summary
                FROM news
                WHERE id = ?
                """,
                (news_id,),
            ).fetchone()
            if row is None:
                continue

            title, link, news_date, created_at, summary = row
            # date 是新聞上的日期，對應作業的文章新增時間。
            published_at = news_date or created_at or ""
            rows.append({
                "title": title,
                "link": link,
                "published_at": published_at,
                "summary": summary or "",
            })
        return rows
    finally:
        conn.close()


def format_context(news_rows):
    blocks = []
    for index, news in enumerate(news_rows, start=1):
        blocks.append(
            f"{index}. 標題：{news['title']}\n"
            f"摘要：{news['summary']}"
        )
    return "\n\n".join(blocks)


def format_sources(news_rows):
    if not news_rows:
        return "\n\n（這次回合沒有重新檢索）"

    lines = ["\n\n參考新聞："]
    for news in news_rows:
        lines.append(f"- 標題：{news['title']}")
        lines.append(f"  連結：{news['link']}")
        lines.append(f"  文章新增時間：{news['published_at']}")
    return "\n".join(lines)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/history")
def get_history():
    return load_history()


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data = request.get_json()
    user_message = (data.get("message") or "").strip()
    # checkbox 決定這次回合要不要檢索。
    use_retrieval = bool(data.get("retrieve"))

    if not user_message:
        return Response("請先輸入問題。", mimetype="text/plain")

    history = load_history()
    messages = build_ollama_messages(history)

    news_rows = []
    if use_retrieval:
        try:
            news_rows = search_news(user_message)
        except Exception as error:
            return Response(
                f"檢索失敗：{error}",
                mimetype="text/plain",
            )

        context = format_context(news_rows)
        messages.append({
            "role": "user",
            "content": (
                f"{user_message}\n\n"
                "請根據以下新聞摘要回答：\n"
                f"{context}"
            ),
        })
    else:
        messages.append({
            "role": "user",
            "content": user_message,
        })

    def generate():
        full_reply = ""

        try:
            stream = client.chat(
                model=MODEL_NAME,
                messages=messages,
                keep_alive="1h",
                stream=True,
            )
        except Exception as error:
            yield f"地端模型沒有回應。請確認 Ollama 已啟動，且已下載 {MODEL_NAME}。\n{error}"
            return

        for part in stream:
            text = part["message"]["content"]
            if text:
                full_reply += text
                yield text

        source_text = format_sources(news_rows if use_retrieval else [])
        full_reply += source_text
        yield source_text

        history.append({
            "role": "user",
            "text": user_message,
        })
        history.append({
            "role": "model",
            "text": full_reply,
        })
        save_history(history)

    return Response(generate(), mimetype="text/plain")


@app.route("/api/clear", methods=["POST"])
def clear_history():
    save_history([])
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(debug=True)
