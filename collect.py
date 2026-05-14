#!/usr/bin/env python3
"""Collecte les données des API AlterVélo et les ajoute dans des fichiers CSV."""

import csv
import os
import json
from datetime import datetime, timezone

import requests

APIS = {
    "vehicle_status": "https://api.gbfs.v3.0.ecovelo.mobi/altervelo/vehicle_status.json",
    "station_status": "https://api.gbfs.v3.0.ecovelo.mobi/altervelo/station_status.json",
}

OUTPUT_DIR = "data"


def fetch_json(url: str) -> dict:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def append_csv(filepath: str, fieldnames: list[str], rows: list[dict]):
    """Ajoute des lignes au CSV, crée le fichier avec en-tête si nécessaire."""
    write_header = not os.path.exists(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def collect_vehicle_status(timestamp: str):
    data = fetch_json(APIS["vehicle_status"])
    vehicles = data["data"]["vehicles"]
    fields = [
        "timestamp", "vehicle_id", "vehicle_type_id", "lat", "lon",
        "current_fuel_percent", "current_range_meters",
        "is_disabled", "is_reserved", "last_reported", "station_id",
    ]
    rows = []
    for v in vehicles:
        rows.append({
            "timestamp": timestamp,
            "vehicle_id": v.get("vehicle_id"),
            "vehicle_type_id": v.get("vehicle_type_id"),
            "lat": v.get("lat"),
            "lon": v.get("lon"),
            "current_fuel_percent": v.get("current_fuel_percent"),
            "current_range_meters": v.get("current_range_meters"),
            "is_disabled": v.get("is_disabled"),
            "is_reserved": v.get("is_reserved"),
            "last_reported": v.get("last_reported"),
            "station_id": v.get("station_id", ""),
        })
    append_csv(os.path.join(OUTPUT_DIR, "vehicle_status.csv"), fields, rows)
    print(f"  vehicle_status: {len(rows)} vélos")


def collect_station_status(timestamp: str):
    data = fetch_json(APIS["station_status"])
    stations = data["data"]["stations"]
    fields = [
        "timestamp", "station_id", "is_installed", "is_renting", "is_returning",
        "num_docks_available", "num_docks_disabled",
        "num_vehicles_available", "num_vehicles_disabled",
        "last_reported", "vehicle_types_available",
    ]
    rows = []
    for s in stations:
        rows.append({
            "timestamp": timestamp,
            "station_id": s.get("station_id"),
            "is_installed": s.get("is_installed"),
            "is_renting": s.get("is_renting"),
            "is_returning": s.get("is_returning"),
            "num_docks_available": s.get("num_docks_available"),
            "num_docks_disabled": s.get("num_docks_disabled"),
            "num_vehicles_available": s.get("num_vehicles_available"),
            "num_vehicles_disabled": s.get("num_vehicles_disabled"),
            "last_reported": s.get("last_reported"),
            "vehicle_types_available": json.dumps(s.get("vehicle_types_available", [])),
        })
    append_csv(os.path.join(OUTPUT_DIR, "station_status.csv"), fields, rows)
    print(f"  station_status: {len(rows)} stations")


def collect_pricing_plans(timestamp: str):
    data = fetch_json(APIS["pricing_plans"])
    plans = data["data"]["plans"]
    fields = [
        "timestamp", "plan_id", "name", "description", "price", "currency",
        "is_taxable", "surge_pricing", "per_min_pricing",
    ]
    rows = []
    for p in plans:
        # Extraire le texte du premier élément de name/description (localisé)
        name = p.get("name", [{}])
        name_text = name[0].get("text", "") if isinstance(name, list) and name else str(name)
        desc = p.get("description", [{}])
        desc_text = desc[0].get("text", "") if isinstance(desc, list) and desc else str(desc)
        rows.append({
            "timestamp": timestamp,
            "plan_id": p.get("plan_id"),
            "name": name_text,
            "description": desc_text,
            "price": p.get("price"),
            "currency": p.get("currency"),
            "is_taxable": p.get("is_taxable"),
            "surge_pricing": p.get("surge_pricing"),
            "per_min_pricing": json.dumps(p.get("per_min_pricing", [])),
        })
    append_csv(os.path.join(OUTPUT_DIR, "pricing_plans.csv"), fields, rows)
    print(f"  pricing_plans: {len(rows)} plans")


def collect_vehicle_types(timestamp: str):
    data = fetch_json(APIS["vehicle_types"])
    vtypes = data["data"]["vehicle_types"]
    fields = [
        "timestamp", "vehicle_type_id", "form_factor",
        "propulsion_type", "max_range_meters",
    ]
    rows = []
    for vt in vtypes:
        rows.append({
            "timestamp": timestamp,
            "vehicle_type_id": vt.get("vehicle_type_id"),
            "form_factor": vt.get("form_factor"),
            "propulsion_type": vt.get("propulsion_type"),
            "max_range_meters": vt.get("max_range_meters"),
        })
    append_csv(os.path.join(OUTPUT_DIR, "vehicle_types.csv"), fields, rows)
    print(f"  vehicle_types: {len(rows)} types")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f"Collecte à {timestamp}")

    collect_vehicle_status(timestamp)
    collect_station_status(timestamp)

    print("Terminé.")


if __name__ == "__main__":
    main()
