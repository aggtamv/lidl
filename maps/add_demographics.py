"""Add area-weighted Eurostat 2021 demographics to Lidl catchments.

Each census-grid value is allocated to a Voronoi polygon according to the
fraction of the 1 km grid cell intersected by that polygon. The results are
proxies, not exact counts, because population is assumed to be distributed
uniformly within each census cell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CATCHMENTS = SCRIPT_DIR / "lidl_catchments_toumba.gpkg"
DEFAULT_GRID = (
    SCRIPT_DIR
    / "demographic_data"
    / "Eurostat_Census-GRID_2021_V3"
    / "Eurostat_Census-GRID_2021_V3"
    / "ESTAT_Census_2021_V3.gpkg"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "lidl_catchments_demographics.gpkg"
GRID_LAYER = "ESTAT_Census_2021_V3"
ANALYSIS_CRS = "EPSG:3035"

# Source field -> descriptive output field.
DEMOGRAPHIC_FIELDS = {
    "T": "population",
    "M": "male_population",
    "F": "female_population",
    "Y_LT15": "population_under_15",
    "Y_1564": "population_15_64",
    "Y_GE65": "population_65_plus",
    "EMP": "employed_population",
    "NAT": "born_in_country",
    "EU_OTH": "born_other_eu",
    "OTH": "born_outside_eu",
    "SAME": "same_residence",
    "CHG_IN": "moved_within_country",
    "CHG_OUT": "moved_from_abroad",
}
SENTINEL_VALUES = {-8888, -9999}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate demographics for Lidl Voronoi catchments using the "
            "Eurostat 2021 one-kilometre census grid."
        )
    )
    parser.add_argument("--catchments", type=Path, default=DEFAULT_CATCHMENTS)
    parser.add_argument(
        "--catchments-layer",
        default="lidl_catchments",
        help="Catchment layer name (default: lidl_catchments).",
    )
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument(
        "--grid-layer",
        default=GRID_LAYER,
        help=f"Census grid layer name (default: {GRID_LAYER}).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--country-code",
        default="EL",
        help="Country code used to filter shared border cells (default: EL).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output GeoPackage if it already exists.",
    )
    return parser.parse_args()


def validate_inputs(
    catchments: gpd.GeoDataFrame, grid_sample: gpd.GeoDataFrame
) -> None:
    """Validate geometry and required attributes before processing."""
    if catchments.empty:
        raise ValueError("The catchment layer is empty.")
    if catchments.crs is None:
        raise ValueError("The catchment layer has no CRS.")
    if "store_index" not in catchments.columns:
        raise ValueError("The catchment layer must contain a store_index field.")
    if catchments["store_index"].duplicated().any():
        raise ValueError("store_index must uniquely identify each catchment.")
    if catchments.geometry.isna().any() or catchments.geometry.is_empty.any():
        raise ValueError("Catchments contain missing or empty geometries.")
    if not set(catchments.geom_type).issubset({"Polygon", "MultiPolygon"}):
        raise ValueError("Catchments must contain polygon geometries.")

    required_grid_fields = {"GRD_ID", "CNTR_ID", *DEMOGRAPHIC_FIELDS}
    missing = required_grid_fields.difference(grid_sample.columns)
    if missing:
        raise ValueError(
            "The census grid is missing required fields: "
            + ", ".join(sorted(missing))
        )
    if grid_sample.crs is None:
        raise ValueError("The census grid has no CRS.")


def load_relevant_grid(
    grid_path: Path,
    grid_layer: str,
    catchments_3035: gpd.GeoDataFrame,
    country_code: str,
) -> gpd.GeoDataFrame:
    """Read only census cells near and intersecting the study area."""
    grid = gpd.read_file(
        grid_path,
        layer=grid_layer,
        bbox=tuple(catchments_3035.total_bounds),
    )
    if grid.crs != catchments_3035.crs:
        grid = grid.to_crs(catchments_3035.crs)

    study_geometry = catchments_3035.geometry.union_all()
    country_pattern = rf"(?:^|-){country_code}(?:-|$)"
    grid = grid.loc[
        grid.geometry.notna()
        & ~grid.geometry.is_empty
        & grid.geometry.intersects(study_geometry)
        & grid["CNTR_ID"].fillna("").str.contains(
            country_pattern, regex=True
        )
    ].copy()
    if grid.empty:
        raise ValueError("No census-grid cells overlap the catchment area.")

    grid["grid_area_m2"] = grid.area
    if (grid["grid_area_m2"] <= 0).any():
        raise ValueError("The census grid contains zero-area geometries.")
    return grid


def allocate_demographics(
    catchments_3035: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Intersect grid cells with catchments and aggregate weighted values."""
    catchment_fields = ["store_index", "geometry"]
    if "store_label" in catchments_3035.columns:
        catchment_fields.insert(1, "store_label")

    grid_fields = [
        "GRD_ID",
        "grid_area_m2",
        *DEMOGRAPHIC_FIELDS.keys(),
        "geometry",
    ]
    pieces = gpd.overlay(
        catchments_3035[catchment_fields],
        grid[grid_fields],
        how="intersection",
        keep_geom_type=True,
    )
    if pieces.empty:
        raise ValueError("The catchment/grid intersection produced no polygons.")

    pieces["intersection_area_m2"] = pieces.area
    pieces["cell_fraction"] = (
        pieces["intersection_area_m2"] / pieces["grid_area_m2"]
    ).clip(0, 1)

    allocated_fields: list[str] = []
    for source_field, output_field in DEMOGRAPHIC_FIELDS.items():
        source_values = pd.to_numeric(pieces[source_field], errors="coerce")
        source_values = source_values.mask(source_values.isin(SENTINEL_VALUES))
        allocated_field = f"allocated_{output_field}"
        pieces[allocated_field] = source_values * pieces["cell_fraction"]
        allocated_fields.append(allocated_field)

    aggregation = {
        field: "sum" for field in allocated_fields
    }
    aggregation.update(
        {
            "intersection_area_m2": "sum",
            "GRD_ID": "nunique",
        }
    )
    summary = pieces.groupby("store_index", as_index=False).agg(aggregation)
    summary = summary.rename(
        columns={
            **{
                f"allocated_{output}": output
                for output in DEMOGRAPHIC_FIELDS.values()
            },
            "GRD_ID": "census_grid_cells",
        }
    )

    enriched = catchments_3035.merge(summary, on="store_index", how="left")
    enriched["catchment_area_km2"] = enriched.area / 1_000_000
    enriched["grid_coverage_ratio"] = (
        enriched["intersection_area_m2"] / enriched.area
    ).clip(0, 1)

    for output_field in DEMOGRAPHIC_FIELDS.values():
        enriched[output_field] = enriched[output_field].round().astype("Int64")

    population = enriched["population"].astype(float)
    working_age = enriched["population_15_64"].astype(float)
    rivals = pd.to_numeric(enriched.get("rival_stores"), errors="coerce")

    enriched["population_density_km2"] = population / enriched["catchment_area_km2"]
    enriched["female_share"] = enriched["female_population"] / population
    enriched["under_15_share"] = enriched["population_under_15"] / population
    enriched["working_age_share"] = enriched["population_15_64"] / population
    enriched["age_65_plus_share"] = enriched["population_65_plus"] / population
    enriched["employment_proxy_rate"] = (
        enriched["employed_population"] / working_age
    )
    enriched["foreign_born_share"] = (
        enriched["born_other_eu"] + enriched["born_outside_eu"]
    ) / population
    enriched["recent_mover_share"] = (
        enriched["moved_within_country"] + enriched["moved_from_abroad"]
    ) / population
    if rivals is not None:
        enriched["population_per_rival"] = population / rivals.replace(0, np.nan)

    enriched["demographic_year"] = 2021
    enriched["demographic_method"] = "1km census grid; area-weighted"
    enriched = enriched.drop(columns="intersection_area_m2")
    return enriched, pieces


def write_output(
    output: Path,
    enriched: gpd.GeoDataFrame,
    pieces: gpd.GeoDataFrame,
    original_crs: object,
    overwrite: bool,
) -> None:
    """Write enriched catchments and auditable intersection pieces."""
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output}. Use --overwrite to replace it."
            )
        output.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    enriched.to_crs(original_crs).to_file(
        output,
        layer="lidl_catchments_demographics",
        driver="GPKG",
    )
    pieces.to_file(
        output,
        layer="demographic_intersections_3035",
        driver="GPKG",
    )


def main() -> None:
    args = parse_args()
    if not args.catchments.exists():
        raise FileNotFoundError(f"Catchment dataset not found: {args.catchments}")
    if not args.grid.exists():
        raise FileNotFoundError(f"Census grid not found: {args.grid}")

    catchments = gpd.read_file(args.catchments, layer=args.catchments_layer)
    grid_sample = gpd.read_file(args.grid, layer=args.grid_layer, rows=1)
    validate_inputs(catchments, grid_sample)

    original_crs = catchments.crs
    catchments_3035 = catchments.to_crs(ANALYSIS_CRS)
    grid = load_relevant_grid(
        args.grid,
        args.grid_layer,
        catchments_3035,
        args.country_code,
    )
    enriched, pieces = allocate_demographics(catchments_3035, grid)
    write_output(args.output, enriched, pieces, original_crs, args.overwrite)

    total_population = int(enriched["population"].sum())
    minimum_coverage = enriched["grid_coverage_ratio"].min()
    print(f"Catchments enriched: {len(enriched)}")
    print(f"Census cells used: {pieces['GRD_ID'].nunique()}")
    print(f"Estimated population: {total_population:,}")
    print(f"Minimum grid coverage: {minimum_coverage:.2%}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
