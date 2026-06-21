"""
Index colormaps + reverse lookup.

The supplied NDVI / NDRE GeoTIFFs are *colorized* RGBA renderings (uint8),
not raw float index rasters. They were produced with the fixed binned color
scales below (identical to ``NDVI_gene.py`` in the project root).

To recover an index value at a pixel we match its RGB to the nearest palette
color and return that bin's midpoint value + human-readable category label.
"""

import numpy as np

# ── NDVI: 12 bins ────────────────────────────────────────────────────────────
NDVI_BINS = [-1.0, -0.2, 0.0, 0.1, 0.2, 0.3, 0.4,
             0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NDVI_HEX = [
    "#a50026",  # -0.2 - 0.0   Bare soil / rock   (color for "< -0.2" #000000 dropped: see note)
    "#d73027",  #  0.0 - 0.1
    "#f46d43",  #  0.1 - 0.2
    "#fdae61",  #  0.2 - 0.3
    "#fee08b",  #  0.3 - 0.4
    "#ffffbf",  #  0.4 - 0.5
    "#d9ef8b",  #  0.5 - 0.6
    "#a6d96a",  #  0.6 - 0.7
    "#66bd63",  #  0.7 - 0.8
    "#1a9850",  #  0.8 - 0.9
    "#006837",  #  0.9 - 1.0   Dense healthy vegetation
]
NDVI_LABELS = [
    "Bare soil (-0.2-0.0)",
    "Very sparse (0.0-0.1)",
    "Sparse (0.1-0.2)",
    "Low (0.2-0.3)",
    "Low-moderate (0.3-0.4)",
    "Moderate (0.4-0.5)",
    "Moderate-good (0.5-0.6)",
    "Good (0.6-0.7)",
    "Healthy (0.7-0.8)",
    "Very healthy (0.8-0.9)",
    "Dense vegetation (0.9-1.0)",
]
# The "#000000 (<-0.2)" bin maps to pure black, which is also generic
# background; we exclude it from the active palette to avoid matching noise.
# Edges that bound the 11 active NDVI colors:
NDVI_ACTIVE_EDGES = NDVI_BINS[1:]   # -0.2 .. 1.0  (12 edges -> 11 bins)

# ── NDRE: 10 bins ────────────────────────────────────────────────────────────
NDRE_BINS = [-1.0, -0.8, -0.6, -0.4, -0.2,
             0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
NDRE_HEX = [
    "#a50026",  # -1.0 - -0.8
    "#d73027",  # -0.8 - -0.6
    "#f46d43",  # -0.6 - -0.4
    "#fdae61",  # -0.4 - -0.2
    "#fee08b",  # -0.2 -  0.0
    "#ffffbf",  #  0.0 -  0.2
    "#d9ef8b",  #  0.2 -  0.4
    "#a6d96a",  #  0.4 -  0.6
    "#1a9850",  #  0.6 -  0.8
    "#006837",  #  0.8 -  1.0   Dense healthy vegetation
]
NDRE_LABELS = [
    "Very low (-1.0--0.8)",
    "Low (-0.8--0.6)",
    "Low-moderate (-0.6--0.4)",
    "Moderate (-0.4--0.2)",
    "Moderate-good (-0.2-0.0)",
    "Good (0.0-0.2)",
    "Healthy (0.2-0.4)",
    "Very healthy (0.4-0.6)",
    "Vigorous (0.6-0.8)",
    "Dense vigorous (0.8-1.0)",
]


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _bin_midpoints(edges):
    return [(edges[i] + edges[i + 1]) / 2.0 for i in range(len(edges) - 1)]


class IndexColormap:
    """Reverse-maps RGB colors back to index values for one vegetation index."""

    def __init__(self, name, hex_colors, edges, labels):
        self.name = name
        self.palette = np.array([_hex_to_rgb(h) for h in hex_colors],
                                dtype=np.float64)          # (n_bins, 3)
        self.midpoints = np.array(_bin_midpoints(edges))   # (n_bins,)
        self.labels = labels
        assert len(self.palette) == len(self.midpoints) == len(self.labels)

    def classify(self, rgb):
        """
        Map an array of RGB triples to (values, labels, distances).

        Parameters
        ----------
        rgb : (N, 3) array of uint8/float RGB.

        Returns
        -------
        values    : (N,) float index midpoints
        labels    : list[str] length N
        distances : (N,) euclidean distance to the matched palette color
                    (useful as a confidence / quality indicator)
        """
        rgb = np.asarray(rgb, dtype=np.float64)
        # distance from every pixel to every palette color -> (N, n_bins)
        d = np.linalg.norm(rgb[:, None, :] - self.palette[None, :, :], axis=2)
        idx = np.argmin(d, axis=1)
        values = self.midpoints[idx]
        labels = [self.labels[i] for i in idx]
        distances = d[np.arange(len(idx)), idx]
        return values, labels, distances

    def category_for_value(self, value):
        """Return the category label whose bin midpoint is nearest ``value``."""
        if value is None or np.isnan(value):
            return None
        i = int(np.argmin(np.abs(self.midpoints - value)))
        return self.labels[i]


NDVI_CMAP = IndexColormap("NDVI", NDVI_HEX, NDVI_ACTIVE_EDGES, NDVI_LABELS)
NDRE_CMAP = IndexColormap("NDRE", NDRE_HEX, NDRE_BINS, NDRE_LABELS)
