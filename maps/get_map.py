#Get map data from OSMnx

#Load Libraries
import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox

OUTPUT_FORMATS = {
    "geojson": {"extension": ".geojson", "driver": "GeoJSON"},
    "gpkg": {"extension": ".gpkg", "driver": "GPKG"},
}

def parse_center_point(value: str) -> tuple[float, float]:
    """Convert 'latitude,longitude' into a coordinate tuple."""
    try:
        latitude, longitude = map(float, value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Center point must use the format LATITUDE,LONGITUDE"
        ) from exc

    if not -90 <= latitude <= 90:
        raise argparse.ArgumentTypeError("Latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise argparse.ArgumentTypeError("Longitude must be between -180 and 180.")

    return latitude, longitude

#Create the argument parser
def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Download map data regarding supermarkets with OSMnx.   "
    )
    parser.add_argument(
        "--query-mode",
        choices=["point", "place"],
        default="place",
        help="Query mode for OSMnx (default: place).",
    ),
    parser.add_argument(
        "--study-area",
        default="Thessaloniki, Greece",
        help="Named place to download map data for (default: Thessaloniki, Greece).",       
        ),
    parser.add_argument(
        "--center-point",
        type=parse_center_point,
        default=(40.6150, 22.9700),  # Toumba, Thessaloniki
        metavar="LATITUDE,LONGITUDE",
        help=(
            "Center point for point query mode as LATITUDE,LONGITUDE "
            "(default: 40.6150,22.9700)."
        ),
    ),
    parser.add_argument(
        "--distance-meters",
        type=int,
        default=5_000,
        help="Distance in meters for point query mode (default: 5000).",
    ),
    parser.add_argument(
        "--output-basename",
        default="supermarkets_thessaloniki",
        help="Base name for output files (default: supermarkets_thessaloniki).",
    ),
    parser.add_argument(
        "--output-format",
        choices=["geojson", "gpkg"],
        default="gpkg",
        help="Geospatial output format (default: gpkg).",
    ),
    
    return parser.parse_args()



class MapDownloader:

    OUTPUT_FORMATS = {
                "geojson": {"extension": ".geojson", "driver": "GeoJSON"},
                "gpkg": {"extension": ".gpkg", "driver": "GPKG"},
            }
    
    def __init__(self, query_mode, study_area, center_point, distance_meters,
                 output_basename, output_format, tags):
        self.query_mode = query_mode
        self.study_area = study_area
        self.center_point = center_point
        self.distance_meters = distance_meters
        self.output_basename = output_basename
        self.output_format = output_format
        self.tags = tags


    def download_supermarkets(self) -> gpd.GeoDataFrame:
        """Return supermarket features inside the configured study area."""
        if self.query_mode == "point":
            stores = ox.features_from_point(
                center_point = self.center_point, tags = self.tags,
                dist = self.distance_meters).reset_index()
        elif self.query_mode == "place":
            stores = ox.features_from_place(
                place = self.study_area, tags = self.tags).reset_index()
        else:
            raise ValueError("QUERY_MODE must be either 'point' or 'place'.")

        stores = stores.rename(columns={"element": "osm_type", "id": "osm_id"})
        stores = stores.to_crs("EPSG:4326")
        stores["geometry"] = stores.geometry.apply(
                lambda geometry: geometry
                if geometry.geom_type == "Point"
                else geometry.representative_point()
            )

        stores["longitude"] = stores.geometry.x
        stores["latitude"] = stores.geometry.y
    
        useful_columns = [
            "osm_type",
            "osm_id",
            "name",
            "brand",
            "operator",
            "shop",
            "addr:street",
            "addr:housenumber",
            "opening_hours",
            "website",
            "phone",
            "longitude",
            "latitude",
            "geometry",
        ]
        return stores[[column for column in useful_columns if column in stores.columns]]


    def save_supermarkets(self,
        supermarkets: gpd.GeoDataFrame, output_format: str) -> Path:
        """Save supermarkets in the requested geospatial format."""
        format_config = OUTPUT_FORMATS[output_format]
        output_file = (
            Path(__file__).parent
            / f"{self.output_basename}{format_config['extension']}"
        )
        write_options = {"driver": format_config["driver"]}
        if output_format == "gpkg":
            write_options["layer"] = "supermarkets"

        supermarkets.to_file(output_file, **write_options)
        return output_file

    def plot_supermarkets(self, supermarkets: gpd.GeoDataFrame) -> None:
        """Plot the supermarkets on a map."""
        if self.query_mode == "point":
            roads = ox.graph_from_point(
                self.center_point, dist=self.distance_meters, network_type="drive"
            )
        elif self.query_mode == "place":
            roads = ox.graph_from_place(self.study_area, network_type="drive")
        else:
            raise ValueError("QUERY_MODE must be either 'point' or 'place'.")
        if self.query_mode == "point":
                area_label = f"({self.distance_meters / 1000:g} km radius)"
                print(f"Downloading data around {self.center_point}: {area_label}")
        else:
            area_label = self.study_area
            print(f"Downloading data for: {area_label}")
        _, ax = ox.plot_graph(
                roads,
                show=False,
                close=False,
                node_size=0,
                edge_color="#999999",
                edge_linewidth=0.6,
                bgcolor="white",
            )
        # Identify Lidl from any available descriptive field. OSM records are not
        # always tagged consistently, so checking name, brand, and operator is safer.
        lidl_mask = supermarkets.index.to_series().isin([])
        for column in ["name", "brand", "operator"]:
            if column in supermarkets.columns:
                lidl_mask = lidl_mask | supermarkets[column].fillna("").str.contains(
                    "lidl", case=False, regex=False
                )
    
        supermarkets.loc[~lidl_mask].plot(
            ax=ax,
            color="red",
            edgecolor="white",
            markersize=35,
            label="Other supermarkets",
            zorder=3,
        )
        supermarkets.loc[lidl_mask].plot(
            ax=ax,
            color="blue",
            edgecolor="white",
            markersize=45,
            label="Lidl",
            zorder=3,
        )
        ax.legend()
        ax.set_title(f"Roads and OSM supermarkets - {area_label}")
        plt.show()

def main() -> None:
    args = parse_args()
    downloader = MapDownloader(
        query_mode=args.query_mode,
        study_area=args.study_area,
        center_point=args.center_point,
        distance_meters=args.distance_meters,
        output_basename=args.output_basename,
        output_format=args.output_format,
        tags={"shop": "supermarket"},
    )
    supermarkets = downloader.download_supermarkets()
    if supermarkets.empty:
        raise RuntimeError("OpenStreetMap returned no supermarkets for the study area.")

    print(f"Found {len(supermarkets)} supermarket features")
    preview_columns = [
        column
        for column in ["osm_type", "osm_id", "name", "brand", "latitude", "longitude"]
        if column in supermarkets.columns
    ]
    print(supermarkets[preview_columns].head(10).to_string(index=False))

    output_file = downloader.save_supermarkets(supermarkets, args.output_format)
    print(f"Saved supermarket points to: {output_file}")

    downloader.plot_supermarkets(supermarkets)

if __name__ == "__main__":
    main()
