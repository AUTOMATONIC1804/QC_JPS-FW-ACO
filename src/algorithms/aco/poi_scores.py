"""
poi_scores.py
-----------------------------------------------------
Enhanced POI scoring system for Quezon City.

- Handles _left/_right attribute logic
- Checks across all major columns
- Prefers left, falls back to right, or picks higher scoring
- Adds 'Classified' (Yes/No) flag
- Includes "update mode" to reclassify unclassified rows only
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import numpy as np

UPDATE_MODE = False  # Set to True to reclassify only unclassified rows


base_dir = Path(r"D:\Quezon_City\data\processed")
input_path = base_dir / "qc_pois_final_scored.geojson"
output_path = base_dir / "qc_pois_final_scored.geojson"

# Left and Right Merge
def merge_lr(row, field_base):
    left_val = str(row.get(f"{field_base}_left", "")).lower()
    right_val = str(row.get(f"{field_base}_right", "")).lower()

    if not left_val and not right_val:
        return ""
    if left_val and not right_val:
        return left_val
    if right_val and not left_val:
        return right_val
    return f"{left_val} {right_val}"  # both exist, combine for keyword detection


def get_all_text_fields(row):
    fields = []

    # 1. Left/right pairs
    for base in ["amenity", "building", "landuse", "shop", "name"]:
        fields.append(merge_lr(row, base))

    # 2. Singles
    singles = [
        "office", "school", "government", "governance_type", "healthcare",
        "public_transport", "bus", "train", "subway", "station", "jeepney",
        "light_rail", "railway", "marketplace", "museum", "leisure", "tourism",
        "historic", "proposed:building", "proposed:station", "proposed:railway",
        "proposed:light_rail"
    ]
    for s in singles:
        if s in row:
            val = str(row.get(s, "")).lower()
            if val:
                fields.append(val)

    return " ".join(fields)


# Categories
def match_keywords(fields, mapping):
    for k, v in mapping.items():
        if k in fields:
            return v
    return 0

def transport_poi(fields):
    mapping = {
        "train": 3, "railway": 3, "station": 3, "mrt": 3, "lrt": 3,
        "bus": 2, "bus_station": 2, "terminal": 2, "subway": 3, "jeepney": 2,
        "tricycle": 1, "public_transport": 2, "transport": 2, "stop_position": 2
    }
    score = match_keywords(fields, mapping)
    return "Transport Facilities", score if score > 0 else 0

def commercial_poi(fields):
    mapping = {
        "mall": 3, "supermarket": 2, "wholesale": 2, "market": 2, "retail": 1,
        "office": 2, "bank": 2, "restaurant": 2, "cafe": 1, "convenience": 1,
        "bar": 1, "hotel": 2, "store": 1, "shop": 1, "furniture": 1, "hardware": 1, "commercial": 2, "electronics": 1
    }
    score = match_keywords(fields, mapping)
    return "Commercial / Offices", score if score > 0 else 0

def health_poi(fields):
    mapping = {
        "hospital": 3, "medical_center": 3, "clinic": 2,
        "pharmacy": 1, "health": 2, "dental": 1, "rehab": 2
    }
    score = match_keywords(fields, mapping)
    return "Health Facilities", score if score > 0 else 0

def education_poi(fields):
    mapping = {
        "university": 3, "college": 2, "school": 1,
        "academy": 1, "training": 1, "kindergarten": 1, "library": 2
    }
    score = match_keywords(fields, mapping)
    return "Education Facilities", score if score > 0 else 0

def recreation_poi(fields):
    mapping = {
        "park": 3, "sports": 2, "sports_centre": 2, "playground": 2,
        "museum": 2, "gym": 2, "cinema": 1, "stadium": 3,
        "resort": 3, "leisure": 2, "recreation": 2
    }
    score = match_keywords(fields, mapping)
    return "Recreational Facilities", score if score > 0 else 0

def government_poi(fields):
    mapping = {
        "city_hall": 3, "barangay_hall": 2, "embassy": 3,
        "courthouse": 2, "police": 2, "fire_station": 2,
        "post_office": 1, "government": 3, "customs": 2,
        "immigration": 2, "governance": 2
    }
    score = match_keywords(fields, mapping)
    return "Government / Institutional", score if score > 0 else 0

def residential_poi(fields):
    mapping = {
        "residential": 2, "apartments": 2, "condo": 2,
        "housing": 2, "dormitory": 1, "subdivision": 2, "village": 2
    }
    score = match_keywords(fields, mapping)
    return "Residential / Housing", score if score > 0 else 0


# Category Scores
CATEGORY_WEIGHTS = {
    "Transport Facilities": 30,
    "Commercial / Offices": 20,
    "Health Facilities": 15,
    "Education Facilities": 10,
    "Recreational Facilities": 10,
    "Residential / Housing": 10,
    "Government / Institutional": 5
}

category_funcs = [
    transport_poi,
    commercial_poi,
    health_poi,
    education_poi,
    recreation_poi,
    government_poi,
    residential_poi,
]

# Classification and Scoring
def classify_and_score(row):
    fields = get_all_text_fields(row)
    best_cat, best_score, best_weighted = "Unclassified", 0, 0

    for func in category_funcs:
        cat, base_score = func(fields)
        if base_score > 0:
            weighted = base_score * (CATEGORY_WEIGHTS[cat] / 10)
            if weighted > best_weighted:
                best_cat, best_score, best_weighted = cat, base_score, weighted

    classified = "Yes" if best_cat != "Unclassified" else "No"
    return best_cat, best_score, best_weighted, classified



if not UPDATE_MODE:
    print("📂 Loading merged POI file...")
    gdf = gpd.read_file(input_path)
    print(f"  → {len(gdf)} features loaded")

    print("🧠 Running full classification...")
    gdf[["Category", "SubScore", "WeightedScore", "Classified"]] = gdf.apply(
        lambda row: pd.Series(classify_and_score(row)), axis=1
    )

    gdf["NormalizedScore"] = gdf["WeightedScore"] / gdf["WeightedScore"].max()

    gdf.to_file(output_path, driver="GeoJSON")
    print(f"✅ Saved updated POI scores to: {output_path}")

else:
    print("\n🔄 Update mode enabled — refreshing unclassified rows...")
    gdf = gpd.read_file(output_path)
    unclassified_mask = gdf["Classified"] == "No"
    num_unclassified = unclassified_mask.sum()
    print(f"  → Found {num_unclassified} unclassified rows")

    if num_unclassified > 0:
        gdf_to_update = gdf[unclassified_mask].copy()
        gdf_to_update[["Category", "SubScore", "WeightedScore", "Classified"]] = gdf_to_update.apply(
            lambda row: pd.Series(classify_and_score(row)), axis=1
        )
        gdf_to_update["NormalizedScore"] = gdf_to_update["WeightedScore"] / gdf_to_update["WeightedScore"].max()
        gdf.update(gdf_to_update)
        gdf.to_file(output_path, driver="GeoJSON")
        print(f"✅ Updated {num_unclassified} previously unclassified features.")
    else:
        print("✅ No unclassified rows left to update.")


summary = gdf["Category"].value_counts().to_frame("Count")
summary["AvgWeighted"] = gdf.groupby("Category")["WeightedScore"].mean()
print("\n📊 Category Summary:")
print(summary.sort_index())
