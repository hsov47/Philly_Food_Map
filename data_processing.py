"""
Philly Food Map — Data Processing
Run once to produce:
  data/philly_food.parquet (restaurants)
  data/philly_nbhd_summary.parquet (neighborhood)
"""

import logging
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIG — edit for specific usage
# ---------------------------------------------------------------------------

# Directory containing your raw Yelp JSON files.
# "." means the same folder as this script.
YELP_DATA_DIR = Path("./raw_data")

# Where to write philly_food.parquet and philly_nbhd_summary.parquet.
# Created automatically if it doesn't exist.
OUTPUT_DIR = Path("data")

# Local path to a Philadelphia neighborhoods GeoJSON file.
GEOJSON_PATH = Path("./raw_data/philadelphia-neighborhoods.geojson")

# Set True to skip reading review.json — much faster for quick testing.
SKIP_REVIEWS = False

# log lines when running
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bounds of Philadelphia 
PHILLY_LAT_MIN, PHILLY_LAT_MAX = 39.85, 40.14
PHILLY_LON_MIN, PHILLY_LON_MAX = -75.28, -74.95

# Cities that count that appear in the Yelp dataset for Philadelphia metro. Claude helped generate
PHILLY_CITIES = {
    "Philadelphia", "Cherry Hill", "Camden", "Bensalem", "Levittown",
    "Abington", "Norristown", "King of Prussia", "Ardmore", "Lansdale",
    "Doylestown", "Willow Grove", "Hatboro", "Jenkintown", "Horsham",
    "West Chester", "Conshohocken", "Phoenixville", "Malvern", "Berwyn",
}

# Allowlist of Yelp food/cuisine tags to keep as cuisine labels. Claude helped generate this list
FOOD_TAGS = {
    # Cuisine types
    "American (Traditional)", "American (New)", "Italian", "Mexican", "Chinese",
    "Japanese", "Korean", "Thai", "Vietnamese", "Indian", "Mediterranean",
    "Greek", "French", "Spanish", "Lebanese", "Turkish", "Ethiopian",
    "Moroccan", "Egyptian", "Persian/Iranian", "Pakistani", "Afghan",
    "Filipino", "Malaysian", "Indonesian", "Singaporean", "Taiwanese",
    "Cantonese", "Szechuan", "Dim Sum", "Shanghainese", "Hong Kong Style Cafe",
    "Latin American", "Brazilian", "Peruvian", "Colombian", "Cuban",
    "Caribbean", "Dominican", "Puerto Rican", "Salvadoran", "Honduran",
    "Guatemalan", "Venezuelan", "Tex-Mex", "Cajun/Creole", "Soul Food",
    "Southern", "Barbeque", "Steakhouses", "Seafood", "Sushi Bars",
    "Ramen", "Poke", "Hawaiian", "Middle Eastern", "Halal", "Kosher",
    "Vegan", "Vegetarian", "Gluten-Free", "Raw Food",
    # Food formats
    "Burgers", "Pizza", "Sandwiches", "Wraps", "Delis", "Diners",
    "Fast Food", "Food Trucks", "Food Stands", "Food Court",
    "Breakfast & Brunch", "Brunch", "Buffets", "Tapas Bars", "Tapas/Small Plates",
    "Gastropubs", "Sports Bars", "Wine Bars", "Cocktail Bars", "Pubs",
    "Breweries", "Beer Bar", "Champagne Bars", "Karaoke", "Dance Clubs",
    "Coffee & Tea", "Cafes", "Bakeries", "Desserts", "Ice Cream & Frozen Yogurt",
    "Bubble Tea", "Juice Bars & Smoothies", "Donuts", "Bagels", "Pretzels",
    "Creperies", "Waffles", "Gelato", "Chocolatiers & Shops", "Candy Stores",
    "Cupcakes", "Patisserie/Cake Shop",
    # Specific dishes/styles
    "Tacos", "Noodles", "Salad", "Soup", "Chicken Wings", "Hot Dogs",
    "Cheesesteaks", "Hoagies", "Pho", "Dumplings", "Empanadas",
    "Fish & Chips", "Fondue", "Comfort Food",
    # Drink focused
    "Bars", "Nightlife", "Wineries", "Distilleries", "Tea Rooms",
}

# Price mapping from Yelp's 1-4 to symbol string
PRICE_MAP = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}

# GeoJSON source — Philadelphia neighborhoods (OpenDataPhilly)
GEOJSON_URL = (
    "https://raw.githubusercontent.com/opendataphilly/odp-data-storage/"
    "master/philadelphia-neighborhoods/philadelphia-neighborhoods.geojson"
)

# ---------------------------------------------------------------------------
# 1. Read business.json
# ---------------------------------------------------------------------------

def load_businesses(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "yelp_academic_dataset_business.json"
    log.info(f"Reading {path} ...")
    df = pd.read_json(path, lines=True)
    log.info(f" Loaded {len(df):,} businesses total")
    return df


# ---------------------------------------------------------------------------
# 2. Filter to Philadelphia restaurants
# ---------------------------------------------------------------------------

def filter_philly_restaurants(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only businesses tagged as restaurants
    is_restaurant = df["categories"].str.contains("Restaurants|Food", na=False)

    # Keep by city name OR by lat/lon box
    in_philly_city = df["city"].isin(PHILLY_CITIES) & (df["state"] == "PA")
    in_philly_box = (
        df["latitude"].between(PHILLY_LAT_MIN, PHILLY_LAT_MAX)
        & df["longitude"].between(PHILLY_LON_MIN, PHILLY_LON_MAX)
    )

    mask = is_restaurant & (in_philly_city | in_philly_box)
    result = df[mask].copy()
    log.info(f"Filtered to {len(result):,} Philadelphia restaurants")

    # Select and rename columns we actually need
    keep = [
        "business_id", "name", "address", "city", "postal_code",
        "latitude", "longitude", "stars", "review_count",
        "is_open", "categories", "hours", "attributes",
    ]
    result = result[[c for c in keep if c in result.columns]].copy()

    # Get price level from attributes
    def extract_price(attrs):
        if isinstance(attrs, dict):
            raw = attrs.get("RestaurantsPriceRange2")
            try:
                return PRICE_MAP.get(int(str(raw).strip("'")), None)
            except (ValueError, TypeError):
                return None
        return None

    result["price"] = result["attributes"].apply(extract_price)
    result.drop(columns=["attributes"], inplace=True)

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Seperate and clean
# ---------------------------------------------------------------------------

def process_categories(df: pd.DataFrame) -> pd.DataFrame:
    # Split the comma-separated string into a list
    df["categories_list"] = (
        df["categories"]
        .fillna("")
        .str.split(", ")
        .apply(lambda tags: [t.strip().title() for t in tags if t.strip()])
    )

    # Primary cuisine is first cuisine tag
    def pick_primary(tags):
        for tag in tags:
            if tag in FOOD_TAGS:
                return tag
        return "Other"

    df["primary_cuisine"] = df["categories_list"].apply(pick_primary)

    # Cuisine tags is only tags in the food list
    df["cuisine_tags"] = df["categories_list"].apply(
        lambda tags: [t for t in tags if t in FOOD_TAGS] or ["Other"]
    )

    df.drop(columns=["categories", "categories_list"], inplace=True)
    log.info("  Categories processed")
    return df


# ---------------------------------------------------------------------------
# 4. Aggregate and join reviews to businesses
# ---------------------------------------------------------------------------

# this was a lot of help from claude
def aggregate_reviews(data_dir: Path, business_ids: set) -> pd.DataFrame:
    path = data_dir / "yelp_academic_dataset_review.json"
    log.info(f"  Reading {path} in chunks ...")

    chunks = []
    chunksize = 100_000

    with tqdm(desc="  Review chunks", unit="chunk") as pbar:
        reader = pd.read_json(path, lines=True, chunksize=chunksize)
        for chunk in reader:
            # Only keep reviews in established list of Philly businesses
            chunk = chunk[chunk["business_id"].isin(business_ids)]
            if not chunk.empty:
                chunks.append(
                    chunk[["business_id", "stars", "date", "useful"]].copy()
                )
            pbar.update(1)

    if not chunks:
        log.warning("  No matching reviews found — check business_ids")
        return pd.DataFrame(columns=["business_id", "review_count_actual",
                                     "recency_score", "useful_avg"])

    reviews = pd.concat(chunks, ignore_index=True)
    reviews["date"] = pd.to_datetime(reviews["date"])

    agg = reviews.groupby("business_id").agg(
        review_count_actual=("stars", "count"),
        useful_avg=("useful", "mean"),
    ).reset_index()

    log.info(f"  Aggregated reviews for {len(agg):,} businesses")
    return agg


# ---------------------------------------------------------------------------
# 5. Spatial join/assign a neighborhood to each restaurant
# ---------------------------------------------------------------------------

# claude helped me figure out how to use a geojson
def load_neighborhoods(geojson_path: Path | None) -> gpd.GeoDataFrame:
    # Name column in philadelphia-neighborhoods.geojson is LISTNAME
    NAME_COL = "LISTNAME"

    
    if geojson_path and geojson_path.exists():
        log.info(f"  Loading neighborhoods from {geojson_path}")
        gdf = gpd.read_file(geojson_path)
    else:
        log.info(f"  Downloading neighborhoods GeoJSON from {GEOJSON_URL}")
        import requests
        r = requests.get(GEOJSON_URL, timeout=30)
        r.raise_for_status()
        gdf = gpd.GeoDataFrame.from_features(
            r.json()["features"], crs="EPSG:4326" # EPSG:4326 is specific coordinate system
        )

    if NAME_COL not in gdf.columns:
        available = [c for c in gdf.columns if c != "geometry"]
        raise ValueError(
            f"Expected column {NAME_COL!r} not found in GeoJSON.\n"
            f"Available columns: {available}\n"
            f"Update NAME_COL inside load_neighborhoods() to the correct column."
        )

    gdf = gdf.rename(columns={NAME_COL: "neighborhood"})
    gdf = gdf[["neighborhood", "geometry"]].copy()

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    log.info(f"  Loaded {len(gdf):,} neighborhood regions")
    log.info(f"  {gdf['neighborhood'].head(5).tolist()}")
    return gdf


def spatial_join(df: pd.DataFrame, neighborhoods: gpd.GeoDataFrame) -> pd.DataFrame:
    geometry = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Primary join for exact point-in-polygon
    joined = gpd.sjoin(gdf, neighborhoods, how="left", predicate="within")

    # Nearest polygon for any on boundary
    null_mask = joined["neighborhood"].isna()
    if null_mask.any():
        log.info(f"  {null_mask.sum()} points used fallback")
        missed = gdf[null_mask].drop(columns=["index_right", "neighborhood"],
                                     errors="ignore")
        fallback = gpd.sjoin_nearest(missed, neighborhoods, how="left")
        joined.loc[null_mask, "neighborhood"] = fallback["neighborhood"].values

    result = pd.DataFrame(joined.drop(columns=["geometry", "index_right"],
                                      errors="ignore"))
    still_null = result["neighborhood"].isna().sum()
    if still_null:
        log.warning(f"  {still_null} restaurants could not be assigned a neighborhood")
        result["neighborhood"] = result["neighborhood"].fillna("Unknown")

    log.info("  Spatial join complete")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Neighborhood-level aggregation for choropleth map
# ---------------------------------------------------------------------------

def build_neighborhood_summary(
    df: pd.DataFrame, neighborhoods: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # Top cuisine per neighborhood is most common primary_cuisine
    def top_cuisine(series):
        counts = series.value_counts()
        return counts.index[0] if not counts.empty else "Unknown"

    summary = (
        df.groupby("neighborhood")
        .agg(
            avg_rating=("stars", "mean"),
            restaurant_count=("business_id", "count"),
            top_cuisine=("primary_cuisine", top_cuisine),
        )
        .reset_index()
    )
    summary["avg_rating"] = summary["avg_rating"].round(2)

    # Price mode per neighborhood
    price_mode = (
        df.dropna(subset=["price"])
        .groupby("neighborhood")["price"]
        .agg(lambda s: s.value_counts().index[0] if not s.empty else None)
        .reset_index()
        .rename(columns={"price": "typical_price"})
    )
    summary = summary.merge(price_mode, on="neighborhood", how="left")

    # Re-attach polygon geometry for Plotly
    nbhd_geo = neighborhoods.rename(columns={"neighborhood": "neighborhood"})
    summary_geo = nbhd_geo.merge(summary, on="neighborhood", how="left")
    summary_geo["restaurant_count"] = summary_geo["restaurant_count"].fillna(0).astype(int)
    summary_geo["avg_rating"] = summary_geo["avg_rating"].fillna(0)

    log.info(f"  Built neighborhood summary: {len(summary_geo):,} regions")
    return summary_geo


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate that the Yelp JSON files actually exist before starting
    business_file = YELP_DATA_DIR / "yelp_academic_dataset_business.json"
    review_file   = YELP_DATA_DIR / "yelp_academic_dataset_review.json"

    if not business_file.exists():
        raise FileNotFoundError(
            f"Could not find business file at:\n  {business_file.resolve()}\n"
            "Update YELP_DATA_DIR in the CONFIG block to the folder containing your Yelp JSON files."
        )
    if not SKIP_REVIEWS and not review_file.exists():
        raise FileNotFoundError(
            f"Could not find review file at:\n  {review_file.resolve()}\n"
            "Fix YELP_DATA_DIR or set SKIP_REVIEWS = True in CONFIG."
        )

    log.info(f"Yelp data dir : {YELP_DATA_DIR.resolve()}")
    log.info(f"Output dir    : {OUTPUT_DIR.resolve()}")

    # 1: Read 
    businesses = load_businesses(YELP_DATA_DIR)

    # 2: Filter
    philly = filter_philly_restaurants(businesses)
    del businesses  # free memory

    # 3. Categorize
    philly = process_categories(philly)

    # 4. Reviews 
    if not SKIP_REVIEWS:
        review_agg = aggregate_reviews(YELP_DATA_DIR, set(philly["business_id"]))
        if review_agg.empty:
            log.warning(
                "  Review aggregation returned no rows. Default review signals to 0."
            )
            philly["useful_avg"] = 0.0
        else:
            philly = philly.merge(review_agg, on="business_id", how="left")
            # After a left merge some restaurants may have no matching reviews
            for col, default in [
                ("useful_avg", 0.0),
                ("review_count_actual", 0),
                ("recent_review_count", 0),
            ]:
                if col in philly.columns:
                    philly[col] = philly[col].fillna(default)
            philly["useful_avg"] = philly["useful_avg"].round(2)
    else:
        log.info("  Skipping review aggregation (SKIP_REVIEWS = True)")
        philly["useful_avg"] = 0.0

    # Hidden gem score: high rating, low review count 
    rc = philly["review_count"].clip(upper=500)
    philly["hidden_gem_score"] = (
        (philly["stars"] / 5) * (1 - rc / rc.max())
    ).round(3)*100

    # 5. Spatial join 
    log.info("Step 5: Spatial join")
    neighborhoods = load_neighborhoods(GEOJSON_PATH)
    philly = spatial_join(philly, neighborhoods)

    # 6. Neighborhood aggregation
    log.info("Step 6: Neighborhood aggregation")
    nbhd_summary = build_neighborhood_summary(philly, neighborhoods)

    # 7. Write outputs
    point_path = OUTPUT_DIR / "philly_food.parquet"
    nbhd_path  = OUTPUT_DIR / "philly_nbhd_summary.parquet"

    philly.to_parquet(point_path, index=False)
    log.info(f"  Wrote {len(philly):,} rows to {point_path}")

    nbhd_summary.to_parquet(nbhd_path, index=False)
    log.info(f"  Wrote {len(nbhd_summary):,} rows to {nbhd_path}")

    log.info("Preprocessing complete.")
    log.info(f"  Restaurant data: {point_path.resolve()}")
    log.info(f"  Neighborhood data: {nbhd_path.resolve()}")

    # Summary of data processed
    print("\n--- Summary ---")
    print(f"  Total Philly restaurants : {len(philly):,}")
    print(f"  Neighborhoods found  : {philly['neighborhood'].nunique()}")
    print(f"  Cuisines found : {philly['primary_cuisine'].nunique()}")
    print(f"  Avg rating : {philly['stars'].mean():.2f}")
    print(f"  Price breakdown :\n{philly['price'].value_counts().to_string()}")


if __name__ == "__main__":
    main()