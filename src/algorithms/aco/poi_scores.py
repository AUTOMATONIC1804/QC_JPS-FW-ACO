from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

# >>> ADDED
import requests
from shapely.geometry import Polygon, LineString
import time
# <<< END ADDED


UPDATE_MODE = True  # Set to True to reclassify only unclassified rows

base_dir = Path(r"D:\Quezon_City\data\processed")
input_path = base_dir / "qc_pois_final_scored.geojson"
output_path = base_dir / "qc_pois_final_scored.geojson"


# ======================================================
# >>> ROBUST Overpass downloader (fixed)
# ======================================================
def download_osm_way(way_id, max_retries=4, backoff=1.0, timeout=30):
    """Download an OSM way and return TWO features:
       1) the polygon/line geometry
       2) its centroid as a POINT feature
    """
    query = f"""
    [out:json];
    way({way_id});
    (._;>;);
    out body;
    """
    url = "https://overpass-api.de/api/interpreter"
    headers = {"User-Agent": "qc_poi_downloader/1.0"}

    attempt = 0
    while attempt <= max_retries:
        try:
            r = requests.post(url, data={"data": query}, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException:
            attempt += 1
            time.sleep(backoff * attempt)
            continue

        if r.status_code != 200:
            attempt += 1
            time.sleep(backoff * attempt)
            continue

        try:
            data = r.json()
        except ValueError:
            attempt += 1
            time.sleep(backoff * attempt)
            continue

        break
    else:
        print(f"❌ Failed to fetch OSM way {way_id}")
        return None

    # parse nodes
    nodes = {}
    way_tags = {}
    way_nodes = []

    for el in data.get("elements", []):
        if el.get("type") == "node":
            nodes[el["id"]] = (el["lon"], el["lat"])
        elif el.get("type") == "way":
            way_tags = el.get("tags", {})
            way_nodes = el.get("nodes", [])

    coords = [nodes[n] for n in way_nodes if n in nodes]
    if not coords:
        print(f"❗ No coords for {way_id}")
        return None

    # Create polygon or line
    if len(coords) >= 4 and coords[0] == coords[-1]:
        geom = Polygon(coords)
    else:
        geom = LineString(coords)

    # Create centroid point
    centroid_geom = geom.centroid

    # Build the POLYGON row
    gdf_polygon = gpd.GeoDataFrame(
        [{
            **way_tags,
            "osm_id": way_id,
            "type": "polygon"
        }],
        geometry=[geom],
        crs="EPSG:4326"
    )

    # Build the CENTROID point row
    gdf_centroid = gpd.GeoDataFrame(
        [{
            "osm_id": f"{way_id}_centroid",
            "type": "centroid"
        }],
        geometry=[centroid_geom],
        crs="EPSG:4326"
    )

    # Return both as a combined GeoDataFrame
    return pd.concat([gdf_polygon, gdf_centroid], ignore_index=True)

# ======================================================



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
    return f"{left_val} {right_val}"


def get_all_text_fields(row):
    fields = []
    for base in ["amenity", "building", "landuse", "shop", "name"]:
        fields.append(merge_lr(row, base))

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


def match_keywords(fields, mapping):
    for k, v in mapping.items():
        if k in fields:
            return v
    return 0


def transport_poi(fields):
    mapping = {
        "train": 3, "light_rail": 3, "railway": 3, "train_station": 3, "mrt": 3, "lrt": 3,
        "bus": 2, "bus_station": 2, "terminal": 2, "subway": 3, "jeepney": 2,
        "tricycle": 1, "public_transport": 2, "transport": 2, "stop_position": 2,
        "transportation": 2
    }
    score = match_keywords(fields, mapping)
    return "Transport Facilities", score if score > 0 else 0


def commercial_poi(fields):
    mapping = {
        "mall": 3, "supermarket": 2, "wholesale": 2, "market": 2, "retail": 1,
        "office": 2, "bank": 2, "restaurant": 2, "cafe": 1, "convenience": 1,
        "bar": 1, "hotel": 2, "store": 1, "shop": 1, "furniture": 1, "hardware": 1,
        "commercial": 2, "electronics": 1
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


CATEGORY_WEIGHTS = {
    "Transport Facilities": 25,
    "Commercial / Offices": 25,
    "Health Facilities": 15,
    "Education Facilities": 15,
    "Recreational Facilities": 10,
    "Government / Institutional": 10
}

category_funcs = [
    transport_poi,
    commercial_poi,
    health_poi,
    education_poi,
    recreation_poi,
    government_poi,
]


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



# ======================================================
# MAIN PROCESS
# ======================================================
if not UPDATE_MODE:
    print("📂 Loading merged POI file...")
    gdf = gpd.read_file(input_path)
    print(f"  → {len(gdf)} features loaded")

    # Insert OSM Way
    print("⬇ Downloading OSM Way 564479403...")
    osm_extra = download_osm_way(402882917)

    if osm_extra is not None:
        print(f"  → Downloaded with {len(osm_extra.columns)} columns")
        gdf = pd.concat([gdf, osm_extra], ignore_index=True)
        print(f"  → New total: {len(gdf)} features")
    else:
        print("❗ Failed to download OSM way — continuing without it.")

    print("🧠 Running full classification...")
    gdf[["Category", "SubScore", "WeightedScore", "Classified"]] = gdf.apply(
        lambda row: pd.Series(classify_and_score(row)), axis=1
    )

    gdf["NormalizedScore"] = gdf["WeightedScore"] / gdf["WeightedScore"].max()

    # Ensure only one geometry column
    for col in gdf.columns:
        if isinstance(gdf[col], gpd.GeoSeries) and col != "geometry":
            print(f"🧹 Removing extra geometry column: {col}")
            gdf = gdf.drop(columns=[col])

    gdf.to_file(output_path, driver="GeoJSON")
    print(f"✅ Saved to: {output_path}")

else:
    print("\n🔄 Update mode: refreshing unclassified rows...")
    gdf = gpd.read_file(output_path)
    unclassified_mask = gdf["Classified"] == "No"

    if unclassified_mask.any():
        gdf_to_update = gdf[unclassified_mask].copy()
        gdf_to_update[["Category", "SubScore", "WeightedScore", "Classified"]] = gdf_to_update.apply(
            lambda row: pd.Series(classify_and_score(row)), axis=1
        )
        gdf_to_update["NormalizedScore"] = gdf_to_update["WeightedScore"] / gdf_to_update["WeightedScore"].max()
        gdf.update(gdf_to_update)
        gdf.to_file(output_path, driver="GeoJSON")
        print("✅ Updated unclassified rows.")
    else:
        print("No unclassified rows.")


summary = gdf["Category"].value_counts().to_frame("Count")
summary["AvgWeighted"] = gdf.groupby("Category")["WeightedScore"].mean()
print("\n📊 Category Summary:")
print(summary.sort_index())



# ======================================================
# Load POIs + Weights for ACO
# ======================================================
def load_pois_and_weights(pois_file=None):
    base_dir = Path(r"D:\Quezon_City\data\processed")
    pois_file = pois_file or base_dir / "qc_pois_final_scored.geojson"

    print(f"📦 Loading POIs from {pois_file}")
    gdf = gpd.read_file(pois_file)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    print(f"✅ Loaded {len(gdf)} POIs.")
    return gdf, CATEGORY_WEIGHTS

