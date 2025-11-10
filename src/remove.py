"""
remove_specific_osm_attributes.py
-----------------------------------------------------
Removes only the specific OSM-style attributes seen in
the provided screenshots from:
    D:\Quezon_City\data\processed\qc_pois_all_matched.geojson

Output:
    D:\Quezon_City\data\processed\qc_pois_all_matched_clean.geojson
-----------------------------------------------------
"""

import geopandas as gpd
from pathlib import Path

# === Paths ===
base_dir = Path(r"D:\Quezon_City\data\processed")
input_path = base_dir / "qc_pois_transportation_clean.geojson" #qc_pois_all_matched.geojson
output_path = base_dir / "qc_pois_transportation_clean.geojson" #qc_pois_all_matched_clean.geojson

# === Load ===
print("📂 Loading file...")
gdf = gpd.read_file(input_path)
print(f"  → {len(gdf)} features")
print(f"  → {len(gdf.columns)} total columns before cleaning")

# === Explicit columns to remove (from screenshots) ===
cols_to_remove = [
    "FIXME", "abandoned", "access", "addr2:housenumber", "addr2:street",
    "addr3:street", "addr4:street", "addr:barangay", "addr:block",
    "addr:borough", "addr:building", "addr:city", "addr:country",
    "addr:district", "addr:floor", "addr:full", "addr:housename",
    "addr:housenumber", "addr:interpolation", "addr:lot", "addr:neighbourhood",
    "addr:place", "addr:postcode", "addr:province", "addr:quarter",
    "addr:region", "addr:state", "addr:street", "addr:street:corner",
    "addr:subdistrict", "addr:suburb", "addr:unit", "addr:village", "addr:zip",
    "admin_level", "agrarian", "air_conditioning", "alt_name", "alt_name:en",
    "animal_boarding", "architect", "area", "atm", "baby_feeding", "backrest",
    "backup_generator", "barrier", "beauty", "beds", "boundary", "branch",
    "brand", "brand:facebook", "brand:instagram", "brand:short",
    "brand:website", "brand:wikidata", "brand:wikipedia", "brand:wikipedia:es",
    "brand:wikipedia:fr", "brand:wikipedia:de", "brand:wikipedia:en",
    "building:colour", "building:entrances", "building:form",
    "building:levels", "building:levels:underground", "building:material",
    "building:min_level", "bureau_de_change", "capacity:persons", "car_parts",
    "car_repair", "charge", "check_date", "check_date:currency:XBT",
    "check_date:opening_hours", "closed", "clothes", "compressed_air",
    "computer:parts", "computer:type", "construction",
    "contact:email", "contact:facebook",
    "contact:instagram", "contact:mobile", "contact:phone", "contact:twitter",
    "contact:website", "contact:youtube", "contributor:ph", "covered", "craft",
    "cuisine", "currency:PHP", "currency:XBT", "dance:style", "delivery",
    "delivery:partner", "denomination", "description", "description:en",
    "description:opening_hours:en", "designation",     "device_charging",
    "device_charging:capacity", "device_charging:fee", "diet:halal", "diet:meat",
    "diet:vegan", "diet:vegetarian", "dine_in", "dispensing",
    "disused", "disused:amenity", "disused:shop","dog",
    "drink:water", "drinking_water", "drive_through", "duplicate",
    "education_program", "ele", "email", "emergency",
    "facebook", "fax","fee", "female", 
    "ferry", "fixme", "floor:material", "foot", 
    "furniture", "garmin:description",
    "strap_line", "strapline", "street_vendor", "stroller",
    "studio", "surface", "surveillance",
    "surveillance:type", "survey:date", "sustenance", "tactile_paving",
    "takeaway", "taxi_vehicle", "tickets:public_transport", "toilets",
    "toilets:wheelchair", "townhall:type", "trade", "training",
    "type", "unisex", "was:amenity", "was:atm",
    "was:name", "website", "website:booking", "website:orders",
    "wheelchair", "wikidata", "wikimedia_commons", "wikipedia",
    "wikipedia:en", "xmas:feature", "geom_type",
    "second_hand", "self_service", "service", "service:bicycle:Bicycle_Sales_and_Service",
    "service:bicycle:cleaning", "service:bicycle:cleaning:charge", "service:bicycle:cleaning:fee", "service:bicycle:diy",
    "service:bicycle:ebike", "service:bicycle:electric_scooters", "service:bicycle:parts", "service:bicycle:pump",
    "service:bicycle:rental", "service:bicycle:repair", "service:bicycle:retail", "service:bicycle:sales",
    "service:bicycle:second_hand", "service:bicycle:service", "service:bicycle:tools", "service:bicycle:wash",
    "service:motorcycle:batteries", "service:vehicle:air_conditioning", "service:vehicle:alignment", "service:vehicle:batteries",
    "service:vehicle:body_repair", "service:vehicle:brakes", "service:vehicle:car_parts", "service:vehicle:car_repair",
    "service:vehicle:diagnostics", "service:vehicle:electrical", "service:vehicle:inspection", "service:vehicle:muffler",
    "service:vehicle:new_car_sales", "service:vehicle:oil_change", "service:vehicle:painting", "service:vehicle:parts",
    "service:vehicle:preventive_maintenance", "service:vehicle:transmission", "service:vehicle:truck_repair", "service:vehicle:tyres",
    "hairdresser", "grades", "healthcare:speciality", "internet_access",
    "ref", "ref:branch", "ref:doh", "ref:vatin",
    "related_law", "religion", "repair", "roof:height",
    "roof:shape", "route_ref", "sale", "service:vehicle:wheels",
    "share_taxi", "shelter", "short_name",
    "smoking", "solar_photovoltaic_panel:sales", "source", "source:date",
    "source:feature", "source:name", "source:name:mapillary", "source:outline",
    "source:position", "source:url", "source_ref", "specialized_education",
    "sport", "start_date", "owner", "ownership",
    "panoramax", "parking", "parking:levels", "parts",
    "payment:bank_transfer", "payment:bdo", "payment:bdo_pay", "payment:bpi",
    "payment:card", "payment:cash", "payment:cards", "payment:coins",
    "payment:contactless", "payment:credit_cards", "payment:debit_cards", "payment:electronic_purses",
    "payment:ep_beep", "payment:gcash", "payment:landbank_pay", "payment:lightning",
    "payment:lightning_contactless", "payment:mastercard", "payment:mastercard_debit", "payment:maya",
    "payment:mobilepay", "payment:nfc_mobile_payments", "payment:notes", "payment:onchain",
    "payment:paymaya", "payment:paypal", "payment:qr_code", "payment:visa",
    "payment:visa_debit", "payment_centre", "pets", "phone",
    "phone:mobile", "population:pupils:2012", "population:pupils:2015", "postal_code",
    "name:abbr", "name:acronym", "name:ar", "name:de",
    "name:en", "name:eo", "name:es", "name:etymology",
    "name:etymology:wikidata", "name:etymology:wikipedia", "name:fil", "name:id",
    "name:it", "name:ja", "name:ja-Latn", "name:ja_kana",
    "name:ja_rm", "name:ms", "name:pl", "name:ru",
    "name:tl", "name:vec", "name:zh", "name:zh-Hant",
    "narrow", "nat_name", "image", "indoor",
    "industrial", "internet_access", "internet_access:fee", "is_in",
    "is_in:city", "is_in:region", "is_in:zip", "isced:level",
    "lane_markings", "laundry_service", "layer",
    "level", "lgbtq", "lit", "loc_name",
    "loc_ref", "location", "male", "man_made",
    "mapillary", "mapillary:map_feature", "massage", "min_height",
    "mobile", "money_transfer", "motorcycle:clothes", "motorcycle:parts",
    "motorcycle:repair", "motorcycle:sales", "official_name", "old_addr:street",
    "old_name", "old_name:en", "old_name:tl", "old_ref",
    "opening_date", "opening_hours", "opening_hours:covid19", "opening_hours:signed",
    "operational_status", "operator", "operator:en", "operator:short",
    "operator:tl", "operator:type", "operator:wikidata", "operator:wikipedia",
    "organic", "origin", "outdoor_seating", "previously",
    "height", "highway", "network",
    "network:wikidata", "nohousenumber", "noname", "not:brand",
    "not:brand:wikidata", "note", "protect_class", "protection_title", "railway:ref"


]

# === Drop if columns exist ===
existing_cols_to_drop = [c for c in cols_to_remove if c in gdf.columns]

print(f"🧹 Found {len(existing_cols_to_drop)} matching columns to remove.")

if existing_cols_to_drop:
    for c in existing_cols_to_drop:
        print(f"   - {c}")
    gdf = gdf.drop(columns=existing_cols_to_drop)
else:
    print("⚠ No matching columns found (maybe already cleaned).")

# === Save cleaned GeoJSON ===
print("\n💾 Saving cleaned file...")
gdf.to_file(output_path, driver="GeoJSON")

print(f"✅ Saved cleaned file: {output_path}")
print(f"📊 Columns after cleaning: {len(gdf.columns)}")
