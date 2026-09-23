"""Calculate an equal-weight Lidl location-potential proxy.

The score is exploratory: it combines demand, accessibility, and inverse
competition with equal component weights. Every indicator inside a component
also receives equal weight. Rank normalization avoids imposing units or
distributional assumptions before commercial performance data are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DEMOGRAPHICS = SCRIPT_DIR / "lidl_catchments_demographics.gpkg"
DEFAULT_SOURCE = SCRIPT_DIR / "lidl_catchments_toumba.gpkg"
DEFAULT_OUTPUT = SCRIPT_DIR / "lidl_location_scores.gpkg"
PROJECTED_CRS = "EPSG:2100"

DEMAND_INDICATORS = (
    "population_score",
    "population_density_score",
    "working_age_score",
    "employment_score",
)
ACCESSIBILITY_INDICATORS = (
    "road_intersections_score",
    "major_road_access_score",
    "bus_access_score",
    "metro_access_score",
)
COMPETITION_INDICATORS = (
    "rivals_per_10k_score",
    "distance_competition_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate an equal-weight Lidl location-potential score."
    )
    parser.add_argument("--demographics", type=Path, default=DEFAULT_DEMOGRAPHICS)
    parser.add_argument(
        "--demographics-layer",
        default="lidl_catchments_demographics",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="GeoPackage containing rival_supermarkets and study_boundary.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--road-buffer-meters",
        type=float,
        default=500,
        help="Radius for counting road intersections (default: 500).",
    )
    parser.add_argument(
        "--transit-buffer-meters",
        type=float,
        default=800,
        help="Radius for counting bus stops (default: 800).",
    )
    parser.add_argument(
        "--competition-radius-meters",
        type=float,
        default=5_000,
        help="Maximum rival distance for competition pressure (default: 5000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output GeoPackage if it exists.",
    )
    return parser.parse_args()


def validate_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def store_points(catchments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reconstruct Lidl point geometries saved during catchment creation."""
    required = {"store_index", "store_x_m", "store_y_m", "store_label"}
    missing = required.difference(catchments.columns)
    if missing:
        raise ValueError(
            "Demographic catchments are missing: " + ", ".join(sorted(missing))
        )
    return gpd.GeoDataFrame(
        catchments[["store_index", "store_label"]].copy(),
        geometry=gpd.points_from_xy(
            catchments["store_x_m"], catchments["store_y_m"]
        ),
        crs=PROJECTED_CRS,
    )


def download_osm_accessibility(
    study_boundary: object,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Download roads and public-transport features for the study area."""
    boundary_wgs84 = gpd.GeoSeries(
        [study_boundary], crs=PROJECTED_CRS
    ).to_crs("EPSG:4326").iloc[0]

    ox.settings.use_cache = True
    ox.settings.cache_folder = SCRIPT_DIR / "cache"
    roads = ox.graph_from_polygon(
        boundary_wgs84,
        network_type="drive",
        simplify=True,
        retain_all=True,
    )
    roads = ox.project_graph(roads, to_crs=PROJECTED_CRS)
    nodes, edges = ox.graph_to_gdfs(roads)

    tags = {
        "highway": "bus_stop",
        "public_transport": "platform",
        "railway": "station",
        "station": "subway",
        "subway": "yes",
    }
    transport = ox.features_from_polygon(boundary_wgs84, tags=tags).reset_index()
    transport = transport.to_crs(PROJECTED_CRS)
    transport.geometry = transport.geometry.apply(
        lambda geometry: geometry
        if geometry.geom_type == "Point"
        else geometry.representative_point()
    )

    highway = transport.get("highway", pd.Series("", index=transport.index))
    public_transport = transport.get(
        "public_transport", pd.Series("", index=transport.index)
    )
    bus = transport.get("bus", pd.Series("", index=transport.index))
    bus_mask = highway.eq("bus_stop") | (
        public_transport.eq("platform") & bus.eq("yes")
    )

    railway = transport.get("railway", pd.Series("", index=transport.index))
    station = transport.get("station", pd.Series("", index=transport.index))
    subway = transport.get("subway", pd.Series("", index=transport.index))
    metro_mask = railway.eq("station") & (
        station.eq("subway") | subway.eq("yes")
    )

    bus_stops = deduplicate_nearby_points(transport.loc[bus_mask].copy(), 20)
    metro_stations = deduplicate_nearby_points(
        transport.loc[metro_mask].copy(), 100
    )
    return nodes, edges, bus_stops, metro_stations


def deduplicate_nearby_points(
    points: gpd.GeoDataFrame, tolerance_meters: float
) -> gpd.GeoDataFrame:
    """Keep one OSM feature per rounded spatial bucket."""
    if points.empty:
        return points
    result = points.copy()
    result["_bucket_x"] = (result.geometry.x / tolerance_meters).round()
    result["_bucket_y"] = (result.geometry.y / tolerance_meters).round()
    result = result.drop_duplicates(["_bucket_x", "_bucket_y"])
    return result.drop(columns=["_bucket_x", "_bucket_y"])


def is_major_road(value: object) -> bool:
    major = {"motorway", "trunk", "primary", "secondary", "tertiary"}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return any(item in major for item in value)
    return value in major


def calculate_accessibility(
    stores: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    bus_stops: gpd.GeoDataFrame,
    metro_stations: gpd.GeoDataFrame,
    road_buffer: float,
    transit_buffer: float,
) -> pd.DataFrame:
    """Calculate store-level road and public-transport accessibility."""
    street_count = pd.to_numeric(
        nodes.get("street_count", pd.Series(0, index=nodes.index)),
        errors="coerce",
    ).fillna(0)
    intersections = nodes.loc[street_count >= 3]
    major_edges = edges.loc[edges["highway"].apply(is_major_road)]

    records = []
    for store in stores.itertuples():
        point = store.geometry
        road_intersections = int(
            intersections.geometry.distance(point).le(road_buffer).sum()
        )
        major_road_distance = (
            float(major_edges.geometry.distance(point).min())
            if not major_edges.empty
            else np.nan
        )
        bus_stop_count = int(
            bus_stops.geometry.distance(point).le(transit_buffer).sum()
        )
        metro_distance = (
            float(metro_stations.geometry.distance(point).min())
            if not metro_stations.empty
            else np.nan
        )
        records.append(
            {
                "store_index": store.store_index,
                "road_intersections_500m": road_intersections,
                "distance_major_road_m": major_road_distance,
                "bus_stops_800m": bus_stop_count,
                "distance_metro_m": metro_distance,
            }
        )
    return pd.DataFrame(records)


def calculate_competition(
    catchments: gpd.GeoDataFrame,
    stores: gpd.GeoDataFrame,
    rivals: gpd.GeoDataFrame,
    radius_meters: float,
) -> pd.DataFrame:
    """Calculate population-adjusted and distance-decayed rival pressure."""
    rivals = rivals.to_crs(PROJECTED_CRS)
    records = []
    catchment_by_store = catchments.set_index("store_index")
    for store in stores.itertuples():
        catchment = catchment_by_store.loc[store.store_index]
        population = float(catchment["population"])
        rival_count = float(catchment["rival_stores"])
        rivals_per_10k = rival_count / population * 10_000 if population else np.nan

        distances = rivals.geometry.distance(store.geometry)
        distances = distances.loc[distances <= radius_meters]
        # One nearby rival contributes almost 1; influence declines smoothly
        # with distance measured in kilometres.
        pressure = float((1 / (1 + distances / 1_000)).sum())
        records.append(
            {
                "store_index": store.store_index,
                "rivals_per_10k": rivals_per_10k,
                "distance_competition_pressure": pressure,
                "rivals_within_5km": int(len(distances)),
            }
        )
    return pd.DataFrame(records)


def percentile_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Normalize an indicator to [0, 1] using average percentile ranks."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0 or numeric.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=values.index)
    numeric = numeric.fillna(numeric.median())
    score = (numeric.rank(method="average") - 1) / (len(numeric) - 1)
    return score if higher_is_better else 1 - score


def calculate_scores(data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize indicators and calculate equal-weight component scores."""
    result = data.copy()
    result["population_score"] = percentile_score(result["population"], True)
    result["population_density_score"] = percentile_score(
        result["population_density_km2"], True
    )
    result["working_age_score"] = percentile_score(
        result["working_age_share"], True
    )
    result["employment_score"] = percentile_score(
        result["employment_proxy_rate"], True
    )
    result["road_intersections_score"] = percentile_score(
        result["road_intersections_500m"], True
    )
    result["major_road_access_score"] = percentile_score(
        result["distance_major_road_m"], False
    )
    result["bus_access_score"] = percentile_score(result["bus_stops_800m"], True)
    result["metro_access_score"] = percentile_score(
        result["distance_metro_m"], False
    )
    result["rivals_per_10k_score"] = percentile_score(
        result["rivals_per_10k"], False
    )
    result["distance_competition_score"] = percentile_score(
        result["distance_competition_pressure"], False
    )

    result["demand_score"] = result[list(DEMAND_INDICATORS)].mean(axis=1)
    result["accessibility_score"] = result[list(ACCESSIBILITY_INDICATORS)].mean(
        axis=1
    )
    result["competition_score"] = result[list(COMPETITION_INDICATORS)].mean(axis=1)
    result["location_potential_score"] = (
        result[["demand_score", "accessibility_score", "competition_score"]]
        .mean(axis=1)
        .mul(100)
    )
    result["location_rank"] = (
        result["location_potential_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    result["score_method"] = "equal components; equal indicators; percentile ranks"
    return result


def write_output(
    output: Path,
    scores: gpd.GeoDataFrame,
    road_edges: gpd.GeoDataFrame,
    bus_stops: gpd.GeoDataFrame,
    metro_stations: gpd.GeoDataFrame,
    overwrite: bool,
) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output exists: {output}. Use --overwrite to replace it."
            )
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_file(output, layer="lidl_location_scores", driver="GPKG")
    road_columns = [
        column
        for column in ["highway", "name", "oneway", "length", "geometry"]
        if column in road_edges.columns
    ]
    roads_to_write = road_edges[road_columns].copy()
    for column in roads_to_write.columns.difference(["geometry"]):
        roads_to_write[column] = roads_to_write[column].apply(
            lambda value: "; ".join(map(str, value))
            if isinstance(value, (list, tuple, set, np.ndarray))
            else value
        )
    roads_to_write.to_file(
        output, layer="road_network", driver="GPKG"
    )
    if not bus_stops.empty:
        bus_stops.to_file(output, layer="bus_stops", driver="GPKG")
    if not metro_stations.empty:
        metro_stations.to_file(output, layer="metro_stations", driver="GPKG")
    scores.drop(columns="geometry").sort_values("location_rank").to_csv(
        output.with_suffix(".csv"), index=False
    )


def main() -> None:
    args = parse_args()
    validate_positive(args.road_buffer_meters, "--road-buffer-meters")
    validate_positive(args.transit_buffer_meters, "--transit-buffer-meters")
    validate_positive(args.competition_radius_meters, "--competition-radius-meters")

    catchments = gpd.read_file(
        args.demographics, layer=args.demographics_layer
    ).to_crs(PROJECTED_CRS)
    rivals = gpd.read_file(args.source, layer="rival_supermarkets")
    boundary = gpd.read_file(args.source, layer="study_boundary").to_crs(
        PROJECTED_CRS
    ).geometry.union_all()
    stores = store_points(catchments)

    nodes, edges, bus_stops, metro_stations = download_osm_accessibility(boundary)
    accessibility = calculate_accessibility(
        stores,
        nodes,
        edges,
        bus_stops,
        metro_stations,
        args.road_buffer_meters,
        args.transit_buffer_meters,
    )
    competition = calculate_competition(
        catchments,
        stores,
        rivals,
        args.competition_radius_meters,
    )
    combined = catchments.merge(accessibility, on="store_index").merge(
        competition, on="store_index"
    )
    scores = calculate_scores(combined)
    write_output(
        args.output,
        scores,
        edges,
        bus_stops,
        metro_stations,
        args.overwrite,
    )

    ranking = scores.sort_values("location_rank")[
        ["location_rank", "store_label", "location_potential_score"]
    ]
    print(ranking.to_string(index=False, formatters={
        "location_potential_score": "{:.2f}".format
    }))
    print(f"Saved: {args.output.resolve()}")
    print(f"Table: {args.output.with_suffix('.csv').resolve()}")


if __name__ == "__main__":
    main()
