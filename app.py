"""
Philly Food Map

Required files:
  data/philly_food.parquet
  data/philly_nbhd_summary.parquet

Note: I've tried running streamlit on the server and it doesn't quite work, I found it's best to run this file locally.
"""

# imports
from pathlib import Path
import colorcet as cc
import datashader as ds
import datashader.transfer_functions as tf
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Page config 
st.set_page_config(
    page_title="Philly Food Map",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Establish Paths to data
DATA_DIR = Path("data")
POINT_PATH = DATA_DIR / "philly_food.parquet"
NBHD_PATH = DATA_DIR / "philly_nbhd_summary.parquet"

MAP_CENTER = {"lat": 39.99, "lon": -75.13}
MAP_ZOOM = 10

COLOR_SCALES = {
    "avg_rating": [[0,"#003B4C"],[0.4,"#004C54"],[0.7,"#00B2A9"],[1.0,"#cc3000"]],
    "restaurant_count": [[0,"#003B4C"],[0.3,"#004C54"],[0.7,"#00B2A9"],[1.0,"#cc3000"]],
    "hidden_gem_score": [[0,"#003B4C"],[0.4,"#004C54"],[0.8,"#00B2A9"],[1.0,"#cc3000"]],
}

CHOROPLETH_LABELS = {
    "avg_rating": "Yelp Rating",
    "restaurant_count": "Restaurant Count",
    "hidden_gem_score": "Hidden Gem Score",
}

# Session state runs once expected keys
def init_state():
    defaults = {
        "selected_nbhd": [],  
        "selected_biz": None,
        "open_choose": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# @st.cache_data Streamlit runs on first load takes from memory for the rest of the session

@st.cache_data
def load_point_data() -> pd.DataFrame:
    df = pd.read_parquet(POINT_PATH)
    # makes sure cusine tags is read as a list
    if df["cuisine_tags"].dtype == object and isinstance(df["cuisine_tags"].iloc[0], str):
        import ast
        df["cuisine_tags"] = df["cuisine_tags"].apply(ast.literal_eval)
    return df


@st.cache_data
def load_nbhd_data() -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(NBHD_PATH)
    if gdf.crs is None: # makes sure the correct coordinate system is used
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


@st.cache_data # generates all possible cuisine tags for filtering
def get_cuisine_options(df: pd.DataFrame) -> list[str]:
    all_tags = df["cuisine_tags"].explode().dropna().unique()
    return sorted(set(all_tags) - {"Other"})


# ---------------------------------------------------------------------------
# Filtering Helper Function
# ---------------------------------------------------------------------------

# consolidates all filters into one list. Claude helped me figure out
def apply_filters(
    df: pd.DataFrame,
    cuisines: list[str],
    min_rating: float,
    prices: list[str],
    open_only: bool,
    min_reviews: int,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if cuisines:
        mask &= df["cuisine_tags"].apply(
            lambda tags: any(c in tags for c in cuisines)
        )
    mask &= df["stars"] >= min_rating
    if prices:
        mask &= df["price"].isin(prices)
    if open_only and "is_open" in df.columns:
        mask &= df["is_open"] == 1
    mask &= df["review_count"] >= min_reviews
    return df[mask].copy()

# ---------------------------------------------------------------------------
# Neighbourhood tooltip
# ---------------------------------------------------------------------------

def compute_nbhd_summary(
    df: pd.DataFrame, nbhd_geo: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    if df.empty:
        return nbhd_geo.copy()

    def top_cuisine(s):
        vc = s.value_counts()
        return vc.index[0] if not vc.empty else "—"

    # reaggrigate stats based on filter change
    agg = (
        df.groupby("neighborhood")
        .agg(
            avg_rating=("stars", "mean"),
            restaurant_count=("business_id", "count"),
            top_cuisine=("primary_cuisine", top_cuisine),
            hidden_gem_score=("hidden_gem_score", "mean"),
        )
        .reset_index()
    )
    agg["avg_rating"]       = agg["avg_rating"].round(2)
    agg["hidden_gem_score"] = agg["hidden_gem_score"].round(3)

    merged = nbhd_geo[["neighborhood", "geometry"]].merge(
        agg, on="neighborhood", how="left"
    )
    merged["restaurant_count"] = merged["restaurant_count"].fillna(0).astype(int)
    merged["avg_rating"]       = merged["avg_rating"].fillna(0)
    return merged

# ---------------------------------------------------------------------------
# Datashader overlay
# ---------------------------------------------------------------------------

def build_datashader_overlay(
    df: pd.DataFrame,
    bounds: tuple,
    width: int = 600,
    height: int = 450,
):
    if df.empty or len(df) < 3:
        return None
    lon_min, lat_min, lon_max, lat_max = bounds
    cvs  = ds.Canvas( #sets up grid of regions based on neighborhood
        plot_width=width, plot_height=height,
        x_range=(lon_min, lon_max), y_range=(lat_min, lat_max),
    )   # how many restraunts per region
    agg  = cvs.points(df, "longitude", "latitude")
    img  = tf.shade(agg, cmap=cc.CET_L4, how="log", min_alpha=40)
    img  = tf.spread(img, px=1) #shade based on count
    rgba = np.array(img.to_pil())
    return go.Image(    #apply to on top of map
        z=rgba,
        x0=lon_min, dx=(lon_max - lon_min) / width,
        y0=lat_max, dy=-(lat_max - lat_min) / height,
        opacity=0.55, hoverinfo="skip"
    )

# ---------------------------------------------------------------------------
# Choropleth
# ---------------------------------------------------------------------------

#plot entire map and shading
def build_choropleth(
    nbhd, color_col, df_filtered, show_datashader, selected_nbhd
):

    #tooltip on neighborhood hover
    geojson = nbhd.__geo_interface__
    hover_text = nbhd.apply(
        lambda r: (
            f"<b>{r['neighborhood']}</b><br>"
            f"Avg rating: {r.get('avg_rating', 0):.2f} ★<br>"
            f"Restaurants: {int(r.get('restaurant_count', 0))}<br>"
            f"Top cuisine: {r.get('top_cuisine', '—')}<br>"
            f"Hidden gem: {r.get('hidden_gem_score', 0):.2f}<br>"
        ),
        axis=1,
    )

    # set up list for selected neighborhoods an use base z value
    sel_list = selected_nbhd or []
    z_base = nbhd[color_col].fillna(0)

    #selected vs unselected line color and width
    line_widths = [
        3.0 if r["neighborhood"] in sel_list else 0.8
        for _, r in nbhd.iterrows()
    ]
    line_colors = [
        "#C8A84B" if r["neighborhood"] in sel_list
        else "rgba(255,255,255,0.4)"
        for _, r in nbhd.iterrows()
    ]

    # draw map using defined above parameters
    fig = go.Figure()
    fig.add_trace(go.Choroplethmapbox(
        geojson=geojson,
        locations=nbhd.index,
        z=z_base,
        colorscale=COLOR_SCALES[color_col],
        marker_opacity=0.65,
        marker_line_width=line_widths,
        marker_line_color=line_colors,
        colorbar=dict(title=CHOROPLETH_LABELS[color_col], thickness=12, len=0.5, x=1.01),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        name="Neighborhoods",
        customdata=nbhd["neighborhood"],
    ))

    # # places the datashader on top of the map
    # if show_datashader and not df_filtered.empty:
    #     ds_trace = build_datashader_overlay(
    #         df_filtered,
    #         bounds=(
    #             df_filtered["longitude"].min() - 0.02,
    #             df_filtered["latitude"].min()  - 0.02,
    #             df_filtered["longitude"].max() + 0.02,
    #             df_filtered["latitude"].max()  + 0.02,
    #         ),
    #     )
    #     if ds_trace:
    #         fig.add_trace(ds_trace)

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=MAP_CENTER,
            zoom=MAP_ZOOM,
            # Lock the map to Philly 
            bounds=dict(
                west=-75.45,   
                east=-74.90,   
                south=39.82,   
                north=40.15,   
            ),

            
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        clickmode="event"
    )

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig

# ---------------------------------------------------------------------------
# Restaurant popup
# ---------------------------------------------------------------------------

# Converts hours from yelp to readable format. Claude
def fmt_hours(raw: str) -> str:
    """Convert Yelp hour string ('10:0-18:0' to '10:00 AM – 6:00 PM')"""
    if not raw or raw == "Closed":
        return "Closed"
    try:
        start, end = raw.split("-")
        def to_12h(t):
            h, m = t.split(":")
            h, m = int(h), int(m)
            suffix = "AM" if h < 12 else "PM"
            h12 = h % 12 or 12
            return f"{h12}:{m:02d} {suffix}"
        return f"{to_12h(start)} – {to_12h(end)}"
    except Exception:
        return raw



@st.dialog("Restaurant details", width="large")
def render_detail_popup(biz: pd.Series):
    # Grabs the row of the selected restraunt
    stars_int = int(biz.get("stars", 0))
    stars_str = "★" * stars_int + "☆" * (5 - stars_int)
    price = biz.get("price", "—") or "—"
    cuisine = biz.get("primary_cuisine", "—")
    nbhd = biz.get("neighborhood", "—")
    address = biz.get("address", "—")
    city = biz.get("city", "Philadelphia")
    reviews = int(biz.get("review_count", 0))
    gem = float(biz.get("hidden_gem_score", 0))
    is_open = biz.get("is_open", None)

    # Header row. Claude helped with the format
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"## {biz['name']}")
        st.markdown(
            f"**{stars_str}** &nbsp; `{biz.get('stars', '—')} / 5.0` &nbsp;"
            f"· &nbsp; {price} &nbsp; · &nbsp; {cuisine}"
        )
        st.markdown(f"**Address:** {address}, {city}")
        st.markdown(f"**Neighbourhood:** {nbhd}")
        if is_open is not None:
            st.markdown("🟢 Currently open" if is_open == 1 else "🔴 Currently closed")
    with col2:
        st.metric("Yelp reviews",   f"{reviews:,}")
        st.metric("Hidden gem", f"{gem:.2f}", help="High rating + low review count = undiscovered local favourite")
    with col3:
        st.metric("Rating",     f"{biz.get('stars', '—')} ★")

    st.markdown("---")

    # Hours
    hours = biz.get("hours")
    if hours and isinstance(hours, dict):
        st.markdown("**Hours**")
        day_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        hours_cols = st.columns(7)
        for i, day in enumerate(day_order):
            hours_cols[i].markdown(f"**{day[:3]}**  \n{fmt_hours(hours.get(day, 'Closed'))}")
        st.markdown("")

    # Cuisine tags
    tags = biz.get("cuisine_tags", [])
    if isinstance(tags, list) and tags:
        st.markdown("**Categories**")
        st.markdown("&nbsp;&nbsp;".join([f"`{t}`" for t in tags[:15]]))

# ---------------------------------------------------------------------------
# Top spots table
# ---------------------------------------------------------------------------

#Helper funtion to returns top 20 
def build_top_table(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    cols = ["name", "neighborhood", "primary_cuisine", "stars", "review_count", "price", "hidden_gem_score", "business_id"]
    available = [c for c in cols if c in df.columns]
    top = (
        df[available]
        .sort_values("stars", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    return top

# ---------------------------------------------------------------------------
# Find me a place popup
# ---------------------------------------------------------------------------

@st.dialog("Find me a place", width="large")
def choose_pick_dialog(df_all, cuisine_options, nbhd_options):
    st.markdown("Set your preferences and we'll randomly pick a spot for you.")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        choose_cuisine = st.selectbox(
            "Cuisine",
            options=["Any"] + cuisine_options,
        )
        choose_nbhd = st.selectbox(
            "Neighbourhood",
            options=["Anywhere"] + nbhd_options,
        )
    with c2:
        choose_min_rating = st.select_slider(
            "Minimum rating",
            options=[1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0],
            value=4.0,
        )
        choose_price = st.multiselect(
            "Price range",
            options=["$", "$$", "$$$", "$$$$"],
            default=["$", "$$", "$$$", "$$$$"],
        )

    st.markdown("---")
    if st.button("Pick one for me!", width='stretch', type="primary"):
        pool = df_all.copy()

        if choose_cuisine != "Any":
            pool = pool[pool["cuisine_tags"].apply(
                lambda tags: choose_cuisine in tags
            )]
        if choose_nbhd != "Anywhere":
            pool = pool[pool["neighborhood"] == choose_nbhd]
        if choose_price:
            pool = pool[pool["price"].isin(choose_price)]
        pool = pool[pool["stars"] >= choose_min_rating]

        if pool.empty:
            st.warning("No restaurants match — try loosening the filters.")
        else:
            pick = pool.sample(1).iloc[0]
            st.session_state.selected_biz = pick["business_id"]
            st.session_state.open_choose      = False
            st.rerun()


# ---------------------------------------------------------------------------
# Sidebar Declared
# ---------------------------------------------------------------------------
def render_sidebar(df, cuisine_options):
    with st.sidebar:
        st.markdown("## 🔔 Philly Food Map")
        st.markdown("---")

        # Restaurant name search
        st.markdown("### Find a restaurant")
        search = st.text_input(
            "Search by name",
            placeholder="e.g. Zahav, Reading Terminal…",
        )
        if search:
            matches = df[df["name"].str.contains(search, case=False, na=False)]
            if matches.empty:
                st.caption("No matches found.")
            else:
                st.caption(f"{len(matches)} result(s)")
                with st.container(height=200):
                    for _, row in matches.iterrows():
                        if st.button(
                            row["name"],
                            key=f"search_btn_{row['business_id']}",
                            width='stretch',
                        ):
                            st.session_state.selected_biz = row["business_id"]
                            st.rerun()

        st.markdown("---")
        st.markdown("### Choropleth mode")
        color_col = st.selectbox(
            "Colour neighbourhoods by",
            options=list(CHOROPLETH_LABELS.keys()),
            format_func=lambda k: CHOROPLETH_LABELS[k],
            label_visibility="collapsed",
        )

        st.markdown("### Filters")

        nbhd_options = sorted(df["neighborhood"].dropna().unique().tolist())
        # Selection is based on all the possible neighborhoods loaded in
        # set by map clicks or sidebar selections
        current_selection = [
            n for n in st.session_state.get("selected_nbhd", [])
            if n in nbhd_options
        ]
        selected_nbhds = st.multiselect(
            "Neighbourhoods",
            options=nbhd_options,
            default=current_selection,
            placeholder="All neighbourhoods",
        )
        # Write back to list and this is the source of truth
        st.session_state.selected_nbhd = selected_nbhds

        cuisines   = st.multiselect("Cuisine", options=cuisine_options,
                                    placeholder="All cuisines")
        min_rating = st.slider("Minimum rating", 1.0, 5.0, 4.0, 0.5)
        prices     = st.multiselect("Price range",
                                    options=["$","$$","$$$","$$$$"],
                                    default=["$","$$","$$$","$$$$"])
        min_reviews = st.slider("Min review count", 0, 500, 10, 10)
        open_only   = st.selectbox("Open status", ["All", "Open now"], index=0) == "Open now"


        st.markdown("---")
        st.caption(f"Dataset: Yelp Open Dataset  \n{len(df):,} Philly restaurants loaded")

    return color_col, cuisines, min_rating, prices, min_reviews, open_only

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# Reruns everthing everytime there's a click
def main():

    #loads the cached data
    init_state()

    if not POINT_PATH.exists() or not NBHD_PATH.exists():
        st.error(
            f"Parquet files not found in `{DATA_DIR}/`.  \n"
            "Run `python preprocess.py` first."
        )
        st.stop()

    df_all = load_point_data()
    nbhd_geo = load_nbhd_data()
    cuisine_opts = get_cuisine_options(df_all)

    # loads data into sidebar selection lists
    color_col, cuisines, min_rating, prices, min_reviews, open_only = (
        render_sidebar(df_all, cuisine_opts)
    )
    show_ds = color_col != "point_density"

    # apply filters from sidebar to everything
    df = apply_filters(df_all, cuisines, min_rating, prices, open_only, min_reviews)
    selected_nbhd = st.session_state.selected_nbhd or []  # always a list
    if selected_nbhd:
        df_table = df[df["neighborhood"].isin(selected_nbhd)].copy()
    else:
        df_table = df.copy()
    nbhd = compute_nbhd_summary(df, nbhd_geo)

    # Dialogs — only one can be open at a time
    if st.session_state.get("open_choose") is True:
        # choose pick takes priority for opening detailed resturant dialog 
        st.session_state.selected_biz = None
        st.session_state.open_choose = False  # reset so reruns don't retrigger dialog
        nbhd_opts = sorted(df_all["neighborhood"].dropna().unique().tolist())
        choose_pick_dialog(df_all, cuisine_opts, nbhd_opts)
    elif st.session_state.selected_biz:
        biz_rows = df_all[df_all["business_id"] == st.session_state.selected_biz]
        if not biz_rows.empty:
            render_detail_popup(biz_rows.iloc[0])
            st.session_state.selected_biz = None

    # Row of overall information of selected filters
    k1, k2, k3, k4, k5 = st.columns([1,1,1,2,2])
    k1.metric("Restaurants", f"{len(df_table):,}")
    k2.metric("Avg rating", f"{df_table['stars'].mean():.2f} ★" if not df_table.empty else "—")
    k3.metric("Neighbourhood", ", ".join(selected_nbhd) if selected_nbhd else "All")
    k4.metric("Cuisines", df_table["primary_cuisine"].nunique() if not df_table.empty else 0)
    k5.metric(
        "Top hidden gem",
        df_table.nlargest(1, "hidden_gem_score")["name"].values[0] if not df_table.empty else "—",
    )

    # appears when filters are applied. Claude addition
    if selected_nbhd:
        nbhd_label = ", ".join(selected_nbhd)
        c_info, c_clear = st.columns([6, 1])
        c_info.info(f"Filtered to **{nbhd_label}** · {len(df_table):,} restaurants")
        if c_clear.button("✕ Clear"):
            st.session_state.selected_nbhd = []
            st.rerun()

    st.markdown("---")

    # Map and top spots table render
    map, top_spots = st.columns([3, 1.2])

    with map:
        if df.empty:
            st.warning("No restaurants match the current filters.")
        else:
            fig = build_choropleth(nbhd, color_col, df, show_ds, selected_nbhd)
            event = st.plotly_chart(
                fig,
                width="stretch",
                config={"scrollZoom": True, "modeBarButtonsToRemove": ["resetViewMapbox"]},
                on_select="rerun",
                selection_mode="points",
                key="choropleth_map",
            )
            # Makes a map click added to neighbourhood filter (claude)
            if event and hasattr(event, "selection") and event.selection:
                pts = event.selection.get("points", [])
                if pts:
                    idx = pts[0].get("point_index")
                    if idx is not None and idx < len(nbhd):
                        clicked = nbhd.iloc[idx]["neighborhood"]
                        current = st.session_state.selected_nbhd or []
                        if clicked in current:
                            current = [n for n in current if n != clicked]  # deselect with another click
                        else:
                            current = current + [clicked]
                        st.session_state.selected_nbhd = current
                        st.rerun()

        st.caption("Click a neighbourhood on the map to filter · Search by name in the sidebar · Click a restaurant for details")

    # Render top spots scrollable
    with top_spots:
        nbhd_heading = ", ".join(selected_nbhd)
        st.markdown(f"#### Top spots{' · ' + nbhd_heading if nbhd_heading else ''}")

        if df_table.empty:
            st.info("No restaurants match the filters.")
        else:
            top_df = build_top_table(df_table)
            st.markdown(
                """<style>
                div[data-testid="stVerticalBlock"] button[kind="secondary"] {
                    text-align: left !important;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                </style>""",
                unsafe_allow_html=True,
            )
            # Scrollable container for the restaurant list
            with st.container(height=480):
                for _, row in top_df.iterrows():
                    if st.button(
                        row["name"],
                        key=f"rest_btn_{row['business_id']}",
                        width='stretch',
                    ):
                        st.session_state.selected_biz = row["business_id"]
                        st.rerun()

    st.markdown("---")

    # choose & Rating distribution 
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Find me a place", width='stretch', type="primary", key="choose_btn"):
            st.session_state.open_choose = True
            st.rerun()

    with c2:
        if not df.empty: #based on selected data
            hist_data = df["stars"].value_counts().sort_index()
            bar_fig   = go.Figure(go.Bar(
                x=hist_data.index.astype(str),
                y=hist_data.values,
                marker_color="#004C54",
            ))
            bar_fig.update_layout(
                title="Rating distribution",
                xaxis_title="Stars", yaxis_title="Count",
                height=300, margin=dict(l=0, r=0, t=36, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6EDF3"),
                xaxis=dict(color="#E6EDF3", gridcolor="#21262D"),
                yaxis=dict(color="#E6EDF3", gridcolor="#21262D"),
            )
            st.plotly_chart(bar_fig, width="stretch")


if __name__ == "__main__":
    main()