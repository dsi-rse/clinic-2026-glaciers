"""Frame footprints from the metadata JSON files, in Antarctic polar stereographic km.

The footprint of a frame is the polygon (four corners, latitude and longitude) where the
satellite took the picture. We project the corners to the same map as the tiles (EPSG:3031).
"""

import json
import math
from pathlib import Path

METRES_PER_KM = 1000

# Antarctic polar stereographic (EPSG:3031): WGS 84, standard parallel 71 degrees south
SEMI_MAJOR_M = 6378137.0
FLATTENING = 1 / 298.257223563
STANDARD_PARALLEL = math.radians(71)


def project(lat: float, lon: float) -> tuple[float, float]:
    """Latitude and longitude in degrees to Antarctic polar stereographic x, y in metres."""
    e = math.sqrt(2 * FLATTENING - FLATTENING**2)

    def t_value(phi: float) -> float:
        factor = ((1 - e * math.sin(phi)) / (1 + e * math.sin(phi))) ** (e / 2)
        return math.tan(math.pi / 4 - phi / 2) / factor

    phi_c = STANDARD_PARALLEL
    m_c = math.cos(phi_c) / math.sqrt(1 - e**2 * math.sin(phi_c) ** 2)
    rho = SEMI_MAJOR_M * m_c * t_value(math.radians(-lat)) / t_value(phi_c)
    longitude = math.radians(lon)
    return rho * math.sin(longitude), rho * math.cos(longitude)


def read_footprint(json_path: Path) -> list[tuple[float, float]]:
    """Corners of a frame as (x, y) in km, read from its metadata JSON."""
    with json_path.open() as file:
        metadata = json.load(file)
    domain = metadata["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]
    points = domain["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
    corners = [project(point["Latitude"], point["Longitude"]) for point in points]
    return [(x / METRES_PER_KM, y / METRES_PER_KM) for x, y in corners]
