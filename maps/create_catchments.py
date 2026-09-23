"""Create proxy Lidl service areas and count rival supermarkets.

The script constructs one bounded Voronoi polygon per Lidl store. A supplied
study-area boundary is preferred; otherwise, it creates a proxy boundary from
the convex hull of every supermarket in the input, buffered by a configurable
distance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import pandas as pd
from shapely import MultiPoint, voronoi_polygons


DEFAULT_INPUT = Path(__file__).with_name("supermarkets_thessaloniki.gpkg")
DEFAULT_OUTPUT = Path(__file__).with_name("lidl_catchments.gpkg")
DEFAULT_PROJECTED_CRS = "EPSG:2100"  # Greek Grid; distances are in metres.
LIDL_FIELDS = ("name", "brand", "operator")
THESSALONIKI_URBAN_MUNICIPALITIES = (
    "Municipality of Thessaloniki, Greece",
    "Municipality of Kalamaria, Greece",
    "Municipality of Neapoli-Sykies, Greece",
    "Municipality of Pavlos Melas, Greece",
    "Municipality of Ampelokipoi-Menemeni, Greece",
    "Municipality of Kordelio-Evosmos, Greece",
    "Municipality of Pylaia-Chortiatis, Greece",
)
OSM_RELATION_OVERRIDES = {
    "Municipality of Thessaloniki, Greece": "R1770680",
    "Municipality of Kalamaria, Greece": "R2348442",
    "Municipality of Neapoli-Sykies, Greece": "R2348444",
    "Municipality of Pavlos Melas, Greece": "R2348445",
    "Municipality of Ampelokipoi-Menemeni, Greece": "R2348441",
    "Municipality of Kordelio-Evosmos, Greece": "R2348443",
    "Municipality of Pylaia-Chortiatis, Greece": "R2348451",
}


def parse_center_point(value: str) -> tuple[float, float]:
    """Parse a command-line latitude,longitude pair."""
    try:
        latitude, longitude = map(float, value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Center point must use LATITUDE,LONGITUDE format."
        ) from exc
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise argparse.ArgumentTypeError("Center-point coordinates are out of range.")
    return latitude, longitude


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create bounded Lidl Voronoi catchments and count non-Lidl "
            "supermarkets in each catchment."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--input-layer",
        default=None,
        help="Input GeoPackage layer (default: first layer).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--boundary",
        type=Path,
        help="Optional polygon dataset used to clip the Voronoi cells.",
    )
    parser.add_argument(
        "--boundary-layer",
        default=None,
        help="Boundary layer when --boundary is a multi-layer dataset.",
    )
    parser.add_argument(
        "--place-boundary",
        action="append",
        help=(
            "Named OSM place whose polygon defines the study area. Repeat this "
            "option to dissolve multiple municipalities into one urban area."
        ),
    )
    parser.add_argument(
        "--thessaloniki-urban-area",
        action="store_true",
        help=(
            "Use the dissolved boundaries of the seven municipalities forming "
            "the Thessaloniki urban core."
        ),
    )
    parser.add_argument(
        "--boundary-buffer-meters",
        type=float,
        default=2_000,
        help=(
            "Buffer around the supermarket convex hull when no boundary is "
            "provided (default: 2000)."
        ),
    )
    parser.add_argument(
        "--center-point",
        type=parse_center_point,
        metavar="LATITUDE,LONGITUDE",
        help=(
            "Use a circular study area around this point. This is useful with "
            "the complete supermarket dataset."
        ),
    )
    parser.add_argument(
        "--radius-meters",
        type=float,
        default=5_500,
        help="Radius for --center-point circular study area (default: 5500).",
    )
    parser.add_argument(
        "--projected-crs",
        default=DEFAULT_PROJECTED_CRS,
        help="Projected CRS used for geometry calculations (default: EPSG:2100).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output GeoPackage if it already exists.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display the catchments over the OSM drive network.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        help="Optional path at which to save the road/catchment plot (for example PNG).",
    )
    return parser.parse_args()


def lidl_mask(stores: gpd.GeoDataFrame) -> pd.Series:
    """Return a Boolean mask matching Lidl in common descriptive OSM fields."""
    mask = pd.Series(False, index=stores.index, dtype=bool)
    for column in LIDL_FIELDS:
        if column in stores.columns:
            mask |= stores[column].fillna("").astype(str).str.contains(
                "lidl", case=False, regex=False
            )
    return mask


def add_store_labels(stores: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add readable, unique labels from available OSM address attributes."""
    result = stores.copy()

    def make_label(store: pd.Series) -> str:
        street = store.get("addr:street")
        number = store.get("addr:housenumber")
        if pd.notna(street) and str(street).strip():
            address = str(street).strip()
            if pd.notna(number) and str(number).strip():
                address += f" {str(number).strip()}"
            return f"Lidl — {address}"
        return f"Lidl — OSM {store.get('osm_id', store.name)}"

    result["store_label"] = result.apply(make_label, axis=1)
    return result


def load_stores(path: Path, layer: str | None) -> gpd.GeoDataFrame:
    """Load, validate, and convert supermarket geometries to points."""
    if not path.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {path}")

    stores = gpd.read_file(path, layer=layer)
    stores = stores.drop_duplicates(
        subset=["osm_type", "osm_id"],
        keep="first",
    ).copy()
    if stores.empty:
        raise ValueError("The input dataset contains no supermarket features.")
    if stores.crs is None:
        raise ValueError("The input dataset has no CRS.")
    if stores.geometry.isna().any() or stores.geometry.is_empty.any():
        raise ValueError("The input contains missing or empty geometries.")

    stores = stores.copy()
    stores.geometry = stores.geometry.apply(
        lambda geometry: geometry
        if geometry.geom_type == "Point"
        else geometry.representative_point()
    )
    return stores


def load_boundary(
    path: Path,
    layer: str | None,
    projected_crs: str,
) -> object:
    """Load and dissolve a polygon boundary into one geometry."""
    if not path.exists():
        raise FileNotFoundError(f"Boundary dataset does not exist: {path}")
    boundary = gpd.read_file(path, layer=layer)
    if boundary.empty or boundary.crs is None:
        raise ValueError("The boundary must contain features and have a CRS.")
    boundary = boundary.to_crs(projected_crs)
    polygon_types = {"Polygon", "MultiPolygon"}
    if not set(boundary.geom_type).issubset(polygon_types):
        raise ValueError("The boundary dataset must contain only polygons.")
    geometry = boundary.geometry.union_all()
    if geometry.is_empty:
        raise ValueError("The dissolved boundary is empty.")
    return geometry


def download_place_boundary(places: list[str], projected_crs: str) -> object:
    """Geocode, validate, project, and dissolve named OSM boundaries."""
    ox.settings.use_cache = True
    ox.settings.cache_folder = Path(__file__).with_name("cache")
    frames = []
    for place in places:
        query = OSM_RELATION_OVERRIDES.get(place, place)
        try:
            frame = ox.geocode_to_gdf(query, by_osmid=query.startswith("R"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Could not obtain a polygon boundary for '{place}' ({query})."
            ) from exc
        frames.append(frame)
    boundary = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), crs=frames[0].crs
    )
    if boundary.empty or boundary.crs is None:
        raise ValueError(f"OSM returned no boundaries for: {', '.join(places)}")
    boundary = boundary.to_crs(projected_crs)
    polygon_types = {"Polygon", "MultiPolygon"}
    if not set(boundary.geom_type).issubset(polygon_types):
        raise ValueError(
            "At least one named-place result is not a polygon. "
            "Use more precise municipality names."
        )
    geometry = boundary.geometry.union_all()
    if geometry.is_empty:
        raise ValueError("OSM returned an empty named-place boundary.")
    return geometry


def create_proxy_boundary(
    stores: gpd.GeoDataFrame, buffer_meters: float
) -> object:
    """Create a fallback boundary around the extent of all supermarkets."""
    if buffer_meters < 0:
        raise ValueError("--boundary-buffer-meters cannot be negative.")
    return stores.geometry.union_all().convex_hull.buffer(buffer_meters)


def create_circular_boundary(
    center_point: tuple[float, float], projected_crs: str, radius_meters: float
) -> object:
    """Create a metric-radius study boundary from latitude/longitude."""
    if radius_meters <= 0:
        raise ValueError("--radius-meters must be positive.")
    latitude, longitude = center_point
    center = gpd.GeoSeries.from_xy(
        [longitude], [latitude], crs="EPSG:4326"
    ).to_crs(projected_crs).iloc[0]
    return center.buffer(radius_meters)


def create_catchments(
    lidl: gpd.GeoDataFrame, boundary: object
) -> gpd.GeoDataFrame:
    """Construct clipped Voronoi polygons and associate each with one Lidl."""
    if len(lidl) < 2:
        raise ValueError("At least two Lidl locations are required for Voronoi cells.")
    if lidl.geometry.duplicated().any():
        duplicates = int(lidl.geometry.duplicated(keep=False).sum())
        raise ValueError(
            f"Found {duplicates} Lidl records at duplicate coordinates; "
            "deduplicate them before constructing Voronoi cells."
        )

    diagram = voronoi_polygons(MultiPoint(lidl.geometry.tolist()), extend_to=boundary)
    cells = [polygon.intersection(boundary) for polygon in diagram.geoms]

    records: list[dict[str, object]] = []
    for source_index, store in lidl.iterrows():
        matching_cells = [cell for cell in cells if cell.covers(store.geometry)]
        if len(matching_cells) != 1:
            raise RuntimeError(
                f"Could not uniquely associate Lidl index {source_index} "
                "with a Voronoi cell."
            )
        record = store.drop(labels="geometry").to_dict()
        record["store_index"] = str(source_index)
        record["store_x_m"] = store.geometry.x
        record["store_y_m"] = store.geometry.y
        record["geometry"] = matching_cells[0]
        records.append(record)

    catchments = gpd.GeoDataFrame(records, geometry="geometry", crs=lidl.crs)
    catchments["area_km2"] = catchments.area / 1_000_000
    return catchments


def count_rivals(
    catchments: gpd.GeoDataFrame, rivals: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Count rival stores in each cell, resolving boundary points by proximity."""
    result = catchments.copy()
    result["rival_stores"] = 0
    if rivals.empty:
        return result

    # Each point is assigned to the nearest Lidl/store_index. For a Voronoi
    # diagram this is equivalent to polygon membership and resolves points that
    # fall exactly on a shared cell boundary without double-counting them.
    lidl_points = gpd.GeoDataFrame(
        catchments[["store_index"]].copy(),
        geometry=gpd.points_from_xy(
            catchments["store_x_m"], catchments["store_y_m"]
        ),
        crs=catchments.crs,
    )

    assigned = gpd.sjoin_nearest(
        rivals,
        lidl_points[["store_index", "geometry"]],
        how="left",
        distance_col="distance_to_lidl_m",
    )
    counts = assigned.groupby("store_index").size()
    result["rival_stores"] = (
        result["store_index"].map(counts).fillna(0).astype(int)
    )
    return result


def write_output(
    output: Path,
    catchments: gpd.GeoDataFrame,
    lidl: gpd.GeoDataFrame,
    rivals: gpd.GeoDataFrame,
    boundary: object,
    boundary_source: str,
    overwrite: bool,
) -> None:
    """Write analysis layers to a GeoPackage."""
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Use --overwrite to replace it."
            )
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    boundary_frame = gpd.GeoDataFrame(
        {"boundary_source": [boundary_source]},
        geometry=[boundary],
        crs=catchments.crs,
    )
    catchments.to_file(output, layer="lidl_catchments", driver="GPKG")
    lidl.to_file(output, layer="lidl_stores", driver="GPKG")
    rivals.to_file(output, layer="rival_supermarkets", driver="GPKG")
    boundary_frame.to_file(output, layer="study_boundary", driver="GPKG")


def plot_catchments(
    catchments: gpd.GeoDataFrame,
    lidl: gpd.GeoDataFrame,
    rivals: gpd.GeoDataFrame,
    boundary: object,
    output: Path | None,
    show: bool,
) -> None:
    """Plot the drivable OSM road network and the catchment analysis layers."""
    boundary_wgs84 = gpd.GeoSeries(
        [boundary], crs=catchments.crs
    ).to_crs("EPSG:4326").iloc[0]

    ox.settings.use_cache = True
    ox.settings.cache_folder = Path(__file__).with_name("cache")
    roads = ox.graph_from_polygon(
        boundary_wgs84,
        network_type="drive",
        simplify=True,
        retain_all=True,
    )
    roads = ox.project_graph(roads, to_crs=catchments.crs)

    _, ax = ox.plot_graph(
        roads,
        show=False,
        close=False,
        node_size=0,
        edge_color="#9a9a9a",
        edge_linewidth=0.45,
        bgcolor="white",
        figsize=(13, 11),
    )
    catchments.plot(
        ax=ax,
        column="rival_stores",
        cmap="YlOrRd",
        alpha=0.28,
        edgecolor="#333333",
        linewidth=1.2,
        legend=True,
        legend_kwds={"label": "Rival supermarkets"},
        zorder=2,
    )
    gpd.GeoSeries([boundary], crs=catchments.crs).plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=1.8,
        zorder=3,
    )
    rivals.plot(
        ax=ax,
        color="#d73027",
        markersize=10,
        alpha=0.65,
        label="Rival supermarket",
        zorder=4,
    )
    lidl.plot(
        ax=ax,
        color="#0050aa",
        edgecolor="white",
        linewidth=0.8,
        markersize=75,
        label="Lidl",
        zorder=5,
    )
    for store in lidl.itertuples():
        ax.annotate(
            store.store_label,
            xy=(store.geometry.x, store.geometry.y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            color="#003b73",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "#0050aa",
                "alpha": 0.82,
                "linewidth": 0.5,
            },
            zorder=6,
        )
    ax.set_title("Lidl proxy catchments, rival supermarkets, and drive network")
    ax.legend(loc="best")
    ax.set_axis_off()
    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Plot saved: {output.resolve()}")
    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    args = parse_args()
    stores = load_stores(args.input, args.input_layer).to_crs(args.projected_crs)

    boundary_options = sum(
        option is not None
        for option in (args.center_point, args.boundary, args.place_boundary)
    ) + int(args.thessaloniki_urban_area)
    if boundary_options > 1:
        raise ValueError(
            "Use only one boundary mode: --center-point, --boundary, "
            "--place-boundary, or --thessaloniki-urban-area."
        )
    if args.center_point:
        boundary = create_circular_boundary(
            args.center_point, args.projected_crs, args.radius_meters
        )
        boundary_source = (
            f"circle: center {args.center_point}, radius {args.radius_meters:g} metres"
        )
    elif args.boundary:
        boundary = load_boundary(
            args.boundary, args.boundary_layer, args.projected_crs
        )
        boundary_source = str(args.boundary.resolve())
    elif args.place_boundary:
        boundary = download_place_boundary(
            args.place_boundary, args.projected_crs
        )
        boundary_source = "OSM named-place boundaries: " + "; ".join(
            args.place_boundary
        )
    elif args.thessaloniki_urban_area:
        places = list(THESSALONIKI_URBAN_MUNICIPALITIES)
        boundary = download_place_boundary(places, args.projected_crs)
        boundary_source = "Thessaloniki urban core: " + "; ".join(places)
    else:
        boundary = create_proxy_boundary(stores, args.boundary_buffer_meters)
        boundary_source = (
            "proxy: supermarket convex hull buffered by "
            f"{args.boundary_buffer_meters:g} metres"
        )

    # Apply the study area before separating Lidl and rivals. This prevents
    # out-of-area stores from affecting cells or competition counts.
    stores = stores.loc[stores.geometry.apply(boundary.covers)].copy()
    is_lidl = lidl_mask(stores)
    lidl = add_store_labels(stores.loc[is_lidl])
    rivals = stores.loc[~is_lidl].copy()
    if lidl.empty:
        raise ValueError("No Lidl stores were identified inside the study area.")

    catchments = create_catchments(lidl, boundary)
    catchments = count_rivals(catchments, rivals)
    write_output(
        args.output,
        catchments,
        lidl,
        rivals,
        boundary,
        boundary_source,
        args.overwrite,
    )

    if args.plot or args.plot_output:
        plot_catchments(
            catchments,
            lidl,
            rivals,
            boundary,
            args.plot_output,
            args.plot,
        )

    print(f"Lidl stores: {len(lidl)}")
    print(f"Rival supermarkets: {len(rivals)}")
    print(f"Catchments created: {len(catchments)}")
    print(f"Boundary: {boundary_source}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
