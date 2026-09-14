import argparse
import csv
import random
import re
import time
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.zameen.com/Homes/Lahore-1-{}.html"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
]


def clean_number(text: str) -> Optional[float]:
    if not text:
        return None
    text = text.replace(",", "").strip().lower()
    match = re.search(r"(\d+(\.\d+)?)", text)
    if not match:
        return None
    value = float(match.group(1))
    if "kanal" in text:
        return value * 4500
    if "marla" in text:
        return value * 225
    if "sq. yd" in text or "sq yd" in text:
        return value * 9
    return value


def parse_price(text: str) -> Optional[float]:
    if not text:
        return None
    raw = text.replace(",", "").strip().lower()
    num = clean_number(raw)
    if num is None:
        return None
    if "arab" in raw:
        return num * 1e9
    if "crore" in raw:
        return num * 1e7
    if "lakh" in raw:
        return num * 1e5
    if "thousand" in raw:
        return num * 1e3
    return num


def get_listing_cards(soup: BeautifulSoup):
    selectors = [
        "article",
        "li[aria-label*='Listing']",
        "div[data-listing-id]",
        "div._45789",
    ]
    cards = []
    for sel in selectors:
        cards = soup.select(sel)
        if cards:
            break
    return cards


def extract_text(card, selectors: List[str]) -> str:
    for selector in selectors:
        node = card.select_one(selector)
        if node and node.get_text(strip=True):
            return node.get_text(" ", strip=True)
    return ""


def parse_card(card) -> Dict[str, Optional[str]]:
    price_text = extract_text(
        card,
        [
            "span[aria-label*='Price']",
            "span[title*='PKR']",
            "div[aria-label*='Price']",
            "span",
        ],
    )
    area_text = extract_text(
        card,
        [
            "span[aria-label*='Area']",
            "span[title*='Marla']",
            "span[title*='Kanal']",
            "span[title*='sq. ft']",
        ],
    )
    beds_text = extract_text(
        card,
        [
            "span[aria-label*='Beds']",
            "span[aria-label*='Bed']",
            "span[title*='Bed']",
        ],
    )
    baths_text = extract_text(
        card,
        [
            "span[aria-label*='Baths']",
            "span[aria-label*='Bath']",
            "span[title*='Bath']",
        ],
    )
    location_text = extract_text(
        card,
        [
            "div[aria-label*='Location']",
            "span[aria-label*='Location']",
            "div[title*='Lahore']",
            "span[title*='Lahore']",
        ],
    )
    property_type_text = extract_text(
        card,
        [
            "span[aria-label*='Type']",
            "span[title*='House']",
            "span[title*='Flat']",
            "span[title*='Plot']",
        ],
    )

    # Fallback: search in whole card text for key signals.
    full_text = card.get_text(" ", strip=True)
    if not price_text:
        price_match = re.search(r"PKR\s*[\d,.]+\s*(Crore|Lakh|Arab)?", full_text, re.I)
        price_text = price_match.group(0) if price_match else ""
    if not area_text:
        area_match = re.search(r"[\d,.]+\s*(Kanal|Marla|sq\.?\s*ft|sq\.?\s*yd)", full_text, re.I)
        area_text = area_match.group(0) if area_match else ""
    if not beds_text:
        bed_match = re.search(r"(\d+)\s*Beds?", full_text, re.I)
        beds_text = bed_match.group(1) if bed_match else ""
    if not baths_text:
        bath_match = re.search(r"(\d+)\s*Baths?", full_text, re.I)
        baths_text = bath_match.group(1) if bath_match else ""
    if not property_type_text:
        type_match = re.search(
            r"\b(House|Flat|Apartment|Plot|Penthouse|Upper Portion|Lower Portion|Farm House)\b",
            full_text,
            re.I,
        )
        property_type_text = type_match.group(1) if type_match else ""
    if not location_text:
        loc_match = re.search(r"([A-Za-z\s\-]+,\s*Lahore)", full_text, re.I)
        location_text = loc_match.group(1) if loc_match else "Lahore"

    if property_type_text:
        property_type_text = property_type_text.strip().title()
        if property_type_text == "Apartment":
            property_type_text = "Flat"

    return {
        "price": parse_price(price_text),
        "area_sqft": clean_number(area_text),
        "bedrooms": int(clean_number(beds_text)) if clean_number(beds_text) is not None else None,
        "bathrooms": int(clean_number(baths_text)) if clean_number(baths_text) is not None else None,
        "city": "Lahore",
        "location": location_text.strip() if location_text else "Lahore",
        "property_type": property_type_text if property_type_text else "",
    }


def scrape_listings(max_pages: int, delay_min: float, delay_max: float) -> List[Dict]:
    session = requests.Session()
    rows = []
    seen = set()

    for page in range(1, max_pages + 1):
        url = BASE_URL.format(page)
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        try:
            response = session.get(url, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"[WARN] page={page} status={response.status_code}")
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            cards = get_listing_cards(soup)
            if not cards:
                print(f"[WARN] no cards on page={page}")
                continue
            print(f"[INFO] page={page}, cards={len(cards)}")
            for card in cards:
                row = parse_card(card)
                key = (
                    row["price"],
                    row["area_sqft"],
                    row["bedrooms"],
                    row["bathrooms"],
                    row["city"],
                    row["location"],
                    row["property_type"],
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
            time.sleep(random.uniform(delay_min, delay_max))
        except Exception as exc:
            print(f"[ERROR] page={page}: {exc}")
    return rows


def save_csv(rows: List[Dict], output_csv: str) -> None:
    fields = [
        "price",
        "area_sqft",
        "bedrooms",
        "bathrooms",
        "city",
        "location",
        "property_type",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Scrape Lahore listings from Zameen.")
    parser.add_argument("--pages", type=int, default=40, help="Number of listing pages to scrape.")
    parser.add_argument("--output", type=str, default="zameen_lahore_raw.csv", help="Output CSV filename.")
    parser.add_argument("--delay-min", type=float, default=1.2, help="Minimum delay between page requests.")
    parser.add_argument("--delay-max", type=float, default=2.8, help="Maximum delay between page requests.")
    args = parser.parse_args()

    print("[INFO] Starting scrape...")
    rows = scrape_listings(args.pages, args.delay_min, args.delay_max)
    save_csv(rows, args.output)
    print(f"[INFO] Saved {len(rows)} rows to {args.output}")
    print("[INFO] Target at least ~450 raw rows to keep >=300 after cleaning.")


if __name__ == "__main__":
    main()
