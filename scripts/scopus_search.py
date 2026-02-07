#!/usr/bin/env python3
"""
Scopus Search Utility per DR-Teacher-CNN Research
"""

import requests
import json
from typing import Optional

API_KEY = "231d7255544959097868bcea0a900234"
BASE_URL = "https://api.elsevier.com/content/search/scopus"

def scopus_search(query: str, count: int = 15, sort: str = "-citedby-count") -> list:
    """
    Cerca su Scopus e ritorna i risultati formattati.

    Args:
        query: Stringa di ricerca
        count: Numero di risultati (default 15)
        sort: Ordinamento (-citedby-count, -coverDate)

    Returns:
        Lista di dict con i paper trovati
    """
    headers = {
        "X-ELS-APIKey": API_KEY,
        "Accept": "application/json"
    }

    params = {
        "query": query,
        "count": count,
        "sort": sort
    }

    try:
        response = requests.get(BASE_URL, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        entries = data.get("search-results", {}).get("entry", [])

        for entry in entries:
            paper = {
                "title": entry.get("dc:title", "N/A"),
                "authors": entry.get("dc:creator", "N/A"),
                "year": entry.get("prism:coverDate", "")[:4] if entry.get("prism:coverDate") else "N/A",
                "journal": entry.get("prism:publicationName", "N/A"),
                "citations": int(entry.get("citedby-count", 0)),
                "doi": entry.get("prism:doi", "N/A"),
                "open_access": entry.get("openaccessFlag", False)
            }
            results.append(paper)

        return results

    except Exception as e:
        print(f"Errore: {e}")
        return []

def print_results(results: list, query: str):
    """Stampa i risultati in formato leggibile."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"Risultati trovati: {len(results)}")
    print('='*80)

    for i, paper in enumerate(results, 1):
        oa = "🔓" if paper["open_access"] else "🔒"
        print(f"\n{i}. [{paper['year']}] {paper['title']}")
        print(f"   Autori: {paper['authors']}")
        print(f"   Journal: {paper['journal']}")
        print(f"   Citazioni: {paper['citations']} | DOI: {paper['doi']} {oa}")

def multi_search(queries: list[str], count_per_query: int = 10):
    """Esegue multiple ricerche."""
    all_results = {}
    for query in queries:
        results = scopus_search(query, count=count_per_query)
        all_results[query] = results
        print_results(results, query)
    return all_results

if __name__ == "__main__":
    # Query prioritarie per il problema F1 Mild
    queries = [
        "mild diabetic retinopathy classification deep learning",
        "class imbalance diabetic retinopathy focal loss",
        "EfficientNet diabetic retinopathy 2023 2024",
    ]

    multi_search(queries, count_per_query=8)
