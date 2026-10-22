# 作業：健康諮詢小幫手

用健康醫療網的新聞做 RAG。問問題時，先把「文章摘要」找回來，再交給地端模型回答。

## 資料從哪裡來

爬蟲在 `crawl_data.py`。它抓的是健康醫療網的一個新聞分類（大腸癌頻道），把標題和全文接在一起，交給 Ollama 寫成摘要，最後存進 `news.db`。

資料表 `news` 對應作業要求的欄位：

| 作業欄位 | 資料表欄位 |
| --- | --- |
| 流水號 | `id` |
| 新聞標題 | `title` |
| 新聞連結 | `link` |
| 文章全文 | `content` |
| 文章摘要 | `summary` |
| 文章新增時間 | `date` |

目前資料庫已經有抓好的新聞，不需要先重爬。

## 執行方法

先啟動 Ollama，並確認有 `gemma4:e4b`。然後在 `cases/RAG/` 執行：

```bash
python make_index.py
python app.py
```

瀏覽器打開 `http://127.0.0.1:5000`。

1. 勾選「多輪對話時進行檢索」，問「大腸癌有可能完全沒有症狀嗎？」
2. 保持勾選，再問「直腸癌治療可以保住排便功能嗎？」
3. 取消勾選，按「整理重點」

每次有檢索的回答後面，都會列出新聞標題、新聞連結、文章新增時間。取消勾選時，模型只根據前面的對話整理，不會再查新聞。

## 向量索引 (Vector Index)

## Windows 環境 - CPU Only 版本
- 安裝建立 C 和 C++ 應用程式所需的元件 (請下載 Community 版本)
  - [連結](https://visualstudio.microsoft.com/zh-hant/vs/features/cplusplus/)
- 使用 FAISS 建立向量索引
  - [安裝說明](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
  - 安裝指令:
    - `pip install faiss-cpu==1.15.1`
    - `conda install pytorch::faiss-cpu==1.14.3`
- 安裝 Sentence Transformers
  - [安裝說明](https://sbert.net/docs/installation.html)
  - 安裝指令:
    - `pip install -U sentence-transformers==6.1.0`
- 範例模型
  - [Pretrained Models](https://sbert.net/docs/sentence_transformer/pretrained_models.html#semantic-search-models)
- 執行程式
  - `python make_index.py`
  - 如果出現 `ImportError: DLL load failed while importing _multiarray_umath: 找不到指定的模組。`
    - 步驟一: 先安裝 numpy 2.x `pip install numpy==2.4.6`
    - 步驟二: 再安裝 numpy 小於 2.0 的版本 `pip install numpy==1.26.4`
    - 步驟三: 重新執行程式 `python make_index.py`
    - 如果還有問題，可以試試安裝 `pip install pybind11==2.12`

## 延伸閱讀
- [A vector index built on TurboQuant, written in Rust with Python bindings](https://github.com/RyanCodrai/turbovec)