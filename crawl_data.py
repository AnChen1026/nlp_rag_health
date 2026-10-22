"""
crawl_data.py

爬取健康醫療網「營養飲食」分類的新聞，並用地端模型（Ollama）產生摘要。

流程：

    列表頁 1 ~ MAX_PAGES
        ↓
    news_list.json（連結、記者、日期）
        ↓
    逐篇抓內頁（標題、全文、圖片）
        ↓
    標題 + 全文 → Ollama → 文章摘要
        ↓
    news_data.json

每抓完一篇就寫回 news_data.json，中途停掉再執行會跳過已完成的文章。
寫完後執行 init_db.py 把資料存進 news.db。
"""

import json
import os
import random
import re
import time

import requests
from bs4 import BeautifulSoup as bs
from ollama import Client


# =============================================================================
# 1. 設定
# =============================================================================

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "gemma4:e4b"

BASE_URL = "https://www.healthnews.com.tw"

# 新聞分類：營養飲食
CHANNEL_NAME = "營養飲食"
CHANNEL_URL = f"{BASE_URL}/channel/f72c8a39-1105-2663-255a-b925ad108bac"

# 每頁 8 篇，15 頁約 120 篇。
MAX_PAGES = 15

LIST_FILE = "news_list.json"
DATA_FILE = "news_data.json"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
}

client = Client(
    host=OLLAMA_HOST,
    timeout=600
)


# =============================================================================
# 2. 摘要
# =============================================================================

def summarize_text(text):
    content = f'''請幫我總結、摘要以下文章內容：

    ============================

    {text}

    ============================

    注意：
    1. 使用繁體中文。
    2. 不要使用 markdown 語法。
    3. 只要提供總結、摘要的結果，不要提供其他內容。
    4. 不需要條列說明，將總結、摘要的結果以完整的文字段落形式呈現，不用換行。
    5. 摘要的字數應該介於 150 到 200 字之間，並且要包含文章的主要重點。

    總結、摘要的結果：'''

    messages = [
        {
            'role': 'system',
            'content': '你是一個專業的新聞總結、摘要專家，請幫我將文章內容進行總結、摘要，並以繁體中文回答。'
        },
        {
            'role': 'user',
            'content': content
        },
    ]

    response = client.chat(
        model=MODEL_NAME,
        messages=messages,
        keep_alive="1h",
        think=False,
        options={
            "temperature": 0.3,
        },
    )

    return response.message.content.strip()


# =============================================================================
# 3. 列表頁：取得每篇新聞的連結、記者、日期
# =============================================================================

def crawl_list():
    li_data = []
    seen_links = set()

    for page in range(1, MAX_PAGES + 1):
        page_url = f"{CHANNEL_URL}/{page}"
        res = requests.get(page_url, headers=headers, timeout=15)

        if res.status_code != 200:
            print(f"列表頁 {page} 失敗：{res.status_code}")
            continue

        soup = bs(res.text, "lxml")

        for li in soup.select("ul.list-unstyled > li.list-item"):
            a = li.select_one("div.list-title > a[href]")
            info = li.select_one("div.list-info")
            if a is None or info is None:
                continue

            link = BASE_URL + a["href"]
            if link in seen_links:
                continue
            seen_links.add(link)

            # 例如：食物營養 健康醫療網／記者黃奕寧報導 2026-10-16
            # 轉載的文章沒有「報導／整理」，例如：營養飲食 桃園電子報 2026-09-22
            # 所以日期和記者分開找。
            source_info = info.get_text(" ", strip=True)

            date = None
            m = re.search(r"\d{4}-\d{1,2}-\d{1,2}", source_info)
            if m:
                date = m.group(0)

            reporter = None
            m = re.search(r"(?:記者)?(\w+)(?:報導|整理)", source_info)
            if m:
                reporter = m.group(1)

            li_data.append({
                "link": link,
                "reporter": reporter,
                "date": date,
            })

        print(f"列表頁 {page}：累計 {len(li_data)} 篇")
        time.sleep(random.randint(1, 3))

    with open(LIST_FILE, "w", encoding="utf-8") as f:
        json.dump(li_data, f, ensure_ascii=False, indent=2)

    return li_data


# =============================================================================
# 4. 內頁：標題、全文、圖片、摘要
# =============================================================================

def crawl_articles(li_data):
    # 讀取已完成的文章，重新執行時跳過。
    done = {}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            for news in json.load(f):
                done[news["link"]] = news

    results = []
    t1 = time.time()

    for index, news in enumerate(li_data):
        if news["link"] in done:
            # 摘要沿用，列表頁的記者、日期用這次抓到的。
            old = done[news["link"]]
            old.update(news)
            results.append(old)
            continue

        res = requests.get(news["link"], headers=headers, timeout=15)
        if res.status_code != 200:
            print(f"內頁失敗：{res.status_code} {news['link']}")
            continue

        soup = bs(res.text, "lxml")

        title_tag = soup.select_one("div#article-title > a[href]")
        content_tag = soup.select_one("div#article-content")
        if title_tag is None or content_tag is None:
            print(f"找不到標題或內文，略過：{news['link']}")
            continue

        img = soup.select_one("div.container-fluid div.col-12.mb-3 > img[src]")

        news["title"] = title_tag.get_text().strip()
        news["content"] = content_tag.get_text().strip()
        news["img_path"] = img["src"] if img else None
        news["summary"] = summarize_text(news["title"] + " " + news["content"])

        results.append(news)

        # 每篇寫一次，中斷後可以接著跑。
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("=" * 80)
        print(f"[{index + 1}/{len(li_data)}] {news['title']}")
        print(f"摘要：{news['summary']}")

        time.sleep(random.randint(2, 5))

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    t2 = time.time()
    print(f"完成 {len(results)} 篇，耗時 {(t2 - t1) / 60:.1f} 分鐘")


if __name__ == "__main__":
    print(f"分類：{CHANNEL_NAME}")
    li_data = crawl_list()
    crawl_articles(li_data)
