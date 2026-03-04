import httpx
import pandas as pd
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# =========================
# Configuration
# =========================
BASE_URL = "http://10.12.1.35:8585"
MODEL_NAME = "Qwen3-4B"
MAX_WORKERS = 10
SUMMARY_MAX_CHARS = 4000
OUTPUT_FILE = "hasil_klasifikasi_news.csv"

VALID_LABELS = {
    "POLITIK_PEMERINTAHAN",
    "EKONOMI_BISNIS",
    "HUKUM_KRIMINAL",
    "OLAHRAGA",
    "TEKNOLOGI_DIGITAL",
    "BENCANA_LINGKUNGAN"
}

SYSTEM_PROMPT = (
    "You are a strict Indonesian news classification system. "
    "Your only task is to read a news article and output exactly one of these 6 labels: "
    "POLITIK_PEMERINTAHAN, EKONOMI_BISNIS, HUKUM_KRIMINAL, OLAHRAGA, TEKNOLOGI_DIGITAL, BENCANA_LINGKUNGAN. "
    "Never output anything else. No explanations, no punctuation, no extra text. Just the label."
)

CLASSIFICATION_PROMPT = """Classify the following Indonesian news article into exactly ONE of these 6 labels:

POLITIK_PEMERINTAHAN
EKONOMI_BISNIS
HUKUM_KRIMINAL
OLAHRAGA
TEKNOLOGI_DIGITAL
BENCANA_LINGKUNGAN

CATEGORY DEFINITIONS:
- POLITIK_PEMERINTAHAN: news about how a country or region is governed and managed through political processes, including government decisions, policies, elections, diplomacy, and state institutions
- EKONOMI_BISNIS: news about the production, distribution, and consumption of goods and services, including financial markets, trade, investment, and business activities
- HUKUM_KRIMINAL: news about the system of rules, enforcement of laws, criminal acts, and the judicial process
- OLAHRAGA: news where the PRIMARY topic is an ongoing or completed sports competition, athlete performance result, or sporting event outcome — not a sports figure's non-sport activity, not sports business, not sports policy
- TEKNOLOGI_DIGITAL: news where the PRIMARY topic is a technology product, digital platform, or tech innovation — not merely mentioning a tool or app in passing
- BENCANA_LINGKUNGAN: news where the PRIMARY topic is a natural disaster (flood, earthquake, landslide, eruption, wildfire) or environmental damage actively occurring — NOT accidents, crimes, or human-caused incidents

DISAMBIGUATION RULES:

POLITIK vs EKONOMI:
- Ask: is the main actor a government/state institution making a political decision? → POLITIK_PEMERINTAHAN
- Ask: is the main actor a company, market, or private sector? → EKONOMI_BISNIS
- Government issues policy or regulation → POLITIK_PEMERINTAHAN
- State budget (APBN/APBD) approval or debate → POLITIK_PEMERINTAHAN
- Company earnings, stock market, private investment, business merger → EKONOMI_BISNIS
- Trade data, inflation, GDP, economic indicators (reported, not decided) → EKONOMI_BISNIS
- Government economic program where the policy decision is the focus → POLITIK_PEMERINTAHAN
- Government economic program where the economic impact/data is the focus → EKONOMI_BISNIS
- News about prices, commodities, supply chain, exports, imports → EKONOMI_BISNIS even if government is mentioned
- News about banks, insurance, fintech, capital markets → EKONOMI_BISNIS even if regulated by government
- News about UMKM, entrepreneurs, cooperatives, industry sectors → EKONOMI_BISNIS

HUKUM_KRIMINAL takes priority over BENCANA_LINGKUNGAN:
- Victims or casualties caused by human action (crime, negligence, accident, violence) → HUKUM_KRIMINAL
- Fire caused by humans or unknown cause → HUKUM_KRIMINAL (not BENCANA)
- Building collapse due to construction fault → HUKUM_KRIMINAL (not BENCANA)
- Traffic accident with casualties → HUKUM_KRIMINAL (not BENCANA)
- Corruption, bribery, fraud, money laundering, KPK, police arrest, court verdict → HUKUM_KRIMINAL (never POLITIK)
- Government official arrested or on trial → HUKUM_KRIMINAL (not POLITIK, even if they are a politician)

BENCANA_LINGKUNGAN only when:
- The natural disaster itself (flood, earthquake, landslide, volcanic eruption, tsunami, wildfire from nature) is the PRIMARY topic
- Includes: scale of damage, number of displaced people, rescue operations in progress
- BMKG weather warning where the weather event is the main focus
- Government response or policy about a disaster → POLITIK_PEMERINTAHAN (not BENCANA)
- Disaster relief budget → EKONOMI_BISNIS (not BENCANA)

OLAHRAGA only when:
- The match, competition, tournament, or athlete performance IS the main topic
- Includes: match results, standings, player transfers, training camp, injury news
- Does NOT include: sports corruption → HUKUM_KRIMINAL, sports funding policy → POLITIK_PEMERINTAHAN

TEKNOLOGI_DIGITAL:
- Tech mentioned only as context → use the more appropriate label instead
- Tech regulation by government → TEKNOLOGI_DIGITAL (not POLITIK)

OUT-OF-SCOPE MAPPING:
- Entertainment, film, music, celebrity → EKONOMI_BISNIS
- Religion, culture, tourism as industry → EKONOMI_BISNIS
- Lifestyle, food, travel (non-policy) → EKONOMI_BISNIS
- Trivia, crossword, quiz, games → TEKNOLOGI_DIGITAL
- Science, research, innovation → TEKNOLOGI_DIGITAL
- Education policy, school regulation, national curriculum → POLITIK_PEMERINTAHAN
- Social welfare policy, poverty alleviation program → POLITIK_PEMERINTAHAN
- Public health policy, hospital regulation, national health program → POLITIK_PEMERINTAHAN

CRITICAL RULES:
- You MUST output ONLY one of the 6 labels above, nothing else.
- Do NOT output any label outside the 6 above under any circumstance.
- Do NOT write explanations, reasoning, or punctuation.
- Even if the article is completely unrelated, you MUST still choose the closest one from the 6 labels.

Title: {title}
Content: {content}

Label:"""


# =========================
# Auth
# =========================
# def get_api_key(username: str, otp: str) -> str:
#     url = f"{BASE_URL}/apikey"
#     with httpx.Client(timeout=10.0) as client:
#         response = client.post(url, json={"username": username, "otp": otp})
#         response.raise_for_status()
#         data = response.json()

#     api_key = data.get("api_key")
#     if not api_key:
#         raise ValueError("API key tidak ditemukan di response.")

#     print("API Key berhasil didapatkan.")
#     return api_key


# =========================
# Dataset
# =========================
def load_test_dataset() -> list[dict]:
    dataset = load_dataset(
        "fahadh4ilyas/indonesian_news_datasets",
        split="test",
        token="hf_xxxxxxx"
    )
    return list(dataset)


# =========================
# Classification
# =========================
def normalize_label(raw: str) -> str | None:
    cleaned = raw.strip().upper().replace(" ", "_").replace("-", "_")
    if cleaned in VALID_LABELS:
        return cleaned
    for label in VALID_LABELS:
        if cleaned in label or label.split("_")[0] == cleaned:
            return label
    return None


def classify_news(title: str, content: str, api_key: str) -> tuple[str | None, str | None]:
    headers = {
        "Authorization": f"Bearer pr-GLv7n8xrOztYJNz-xxhhSIYotbwqEyIMLPs0VKWEb4A",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": CLASSIFICATION_PROMPT.format(title=title, content=content)}
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 20,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    with httpx.Client(timeout=60.0) as client:
        response = client.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    raw_label = data["choices"][0]["message"]["content"].strip()
    return normalize_label(raw_label), data.get("id")


def process_row(row: dict, api_key: str) -> dict:
    try:
        predicted_label, response_id = classify_news(
            title=row["title"],
            content=row["content"][:SUMMARY_MAX_CHARS],
            api_key=api_key
        )
        return {**row, "predicted_label": predicted_label, "response_id": response_id, "error": None}
    except Exception as e:
        return {**row, "predicted_label": None, "response_id": None, "error": str(e)}


# =========================
# Batch Processing
# =========================
def run_batch(rows: list[dict], api_key: str) -> list[dict]:
    total = len(rows)
    results = [None] * total
    counter = {"done": 0}
    lock = Lock()

    print(f"Memproses {total} artikel dengan {MAX_WORKERS} workers...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process_row, row, api_key): idx
            for idx, row in enumerate(rows)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            result = future.result()
            results[idx] = result

            with lock:
                counter["done"] += 1
                done = counter["done"]

            status = f"[ERROR: {result['error']}]" if result["error"] else result["predicted_label"]
            print(f"({done}/{total}) ID: {result['id']} → {status}")

    return results


# =========================
# Main
# =========================
def main():
    api_key = "pr-GLv7n8xrOztYJNz-xxhhSIYotbwqEyIMLPs0VKWEb4A"
    rows = load_test_dataset()
    results = run_batch(rows, api_key)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)

    total_success = df["predicted_label"].notna().sum()
    total_error = df["error"].notna().sum()
    print(f"\nSelesai. Sukses: {total_success} | Error: {total_error}")
    print(f"File disimpan sebagai {OUTPUT_FILE}")


if __name__ == "__main__":
    main()