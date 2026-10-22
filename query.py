'''
匯入套件
'''
import os

# macOS 上 torch 和 faiss 各自帶一份 OpenMP，多執行緒時會互相衝突而 segfault。
# 要在 import 之前設定。
os.environ.setdefault("OMP_NUM_THREADS", "1")

from sentence_transformers import SentenceTransformer
import faiss
import sqlite3

# 基本設定
model_name = 'BAAI/bge-m3'
encoder = SentenceTransformer(model_name)

# 讀取索引
index_path = './vector.index'
index = faiss.read_index(index_path)

# 查詢句子
list_query = ['吃到苯駢芘超標的油品該怎麼辦？']

# 將查詢句子轉換成向量
embeddings = encoder.encode(
    list_query, 
    batch_size=3, 
    show_progress_bar=False,
    normalize_embeddings=True
)

# 查詢
D, I = index.search(embeddings, k=3)

# 顯示結果
list_scores = D.tolist()
list_ids = I.tolist()
print(f"相似度: {list_scores}")
print(f"檢索的 Document IDs 為: {list_ids}")

# 透過 Document ID 查詢對應的新聞標題與內容
conn = sqlite3.connect('./news.db')
try:
    # 建立一個空字串，用來存放使用者查詢的問題與對應的新聞標題與內容
    user_prompt = ''

    # 將新聞標題與內容組合成字串，並加入使用者查詢的問題
    for query, ids, scores in zip(list_query, list_ids, list_scores):
        user_prompt = "=" * 80
        user_prompt += f"\n使用者查詢的問題: {query}"

        # 透過 Document ID 查詢對應的新聞標題與內容
        for doc_id, score in zip(ids, scores):
            user_prompt += f"\n{'-' * 80}"
            user_prompt += f"\nDocument ID: {doc_id}"
            user_prompt += f"\n相似度: {score}"

            # 透過 Document ID 查詢對應的新聞標題與內容
            row = conn.execute(
                'SELECT title, link, date, summary FROM news WHERE id = ?',
                (int(doc_id),)
            ).fetchone()

            title, link, news_date, summary = row
            user_prompt += f"\n標題: {title}"
            user_prompt += f"\n連結: {link}"
            user_prompt += f"\n文章新增時間: {news_date}"
            user_prompt += f"\n文件摘要: {summary}"
    
    # 檢視 user prompt
    print(user_prompt)
finally:
    conn.close()

