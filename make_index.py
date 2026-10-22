import sqlite3
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

'''
[faiss.IndexFlatL2]
使用歐氏距離

[faiss.IndexFlatIP]
IP = Inner Product，
測試使用 Inner Product 來比較 features 資料
同時進行 feature normalization
等同於 cosine similarity
'''

# 模型名稱
model_name = 'BAAI/bge-m3'

# 索引存放路徑
index_path = './vector.index'

# 讀取 model
encoder = SentenceTransformer(model_name)

# 讀取資料庫中的文章摘要，轉成向量
docs = []
doc_ids = []
conn = sqlite3.connect('./news.db')
cursor = conn.cursor()
try:
    # 作業要求把「文章摘要」轉成向量，不是把全文轉成向量。
    rows = cursor.execute(
        "SELECT id, summary FROM news "
        "WHERE summary IS NOT NULL AND summary != ''"
    )

    for news_id, summary in rows:
        docs.append(summary)
        doc_ids.append(news_id)
finally:
    conn.close()

# 將 doc_ids 轉成 numpy array，並指定資料型態為 int64
doc_ids = np.array(doc_ids, dtype='int64')

# 將所有句子轉換成向量，同時計算轉向量時間
embeddings = encoder.encode(
    docs, 
    batch_size=8,
    show_progress_bar=True,
    normalize_embeddings=True
)

# 每次都重建索引。
# 如果沿用舊檔再往裡加，同一篇新聞會被重複放進去。
print(embeddings.shape)
print(len(doc_ids))
dims = embeddings.shape[1]
index = faiss.IndexFlatIP(dims)
index = faiss.IndexIDMap(index)
index.add_with_ids(embeddings, doc_ids)

# 儲存索引
faiss.write_index(index, index_path)
print("索引已儲存到", index_path)