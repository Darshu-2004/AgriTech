-- ============================================================
-- schema.sql
-- PostGIS schema for plant NDVI health data
-- Run once to set up the database.
-- ============================================================

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- ── Main plants table ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS plants (
    id              SERIAL PRIMARY KEY,

    -- identification
    plant_id        TEXT,
    geo_hash        TEXT,
    sector_id       INTEGER,
    sector_label    TEXT,
    row_index       INTEGER,
    col_index       INTEGER,

    -- detection metadata
    pixel_x         INTEGER,
    pixel_y         INTEGER,
    area_px         NUMERIC,
    confidence      NUMERIC,

    -- NDVI health
    ndvi            NUMERIC(8, 6),          -- NDVI proxy value (-1 to +1)
    health_status   TEXT CHECK (health_status IN ('healthy', 'moderate', 'diseased')),
    health_color    TEXT,                   -- hex colour code
    raster_crs      TEXT,                   -- e.g. EPSG:32647

    -- geometry (WGS84 stored for compatibility; EPSG:4326)
    geom            GEOMETRY(Point, 4326),

    -- audit
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Spatial index (required for efficient geo queries)
CREATE INDEX IF NOT EXISTS idx_plants_geom
    ON plants USING GIST (geom);

-- Fast look-ups by health
CREATE INDEX IF NOT EXISTS idx_plants_health_status
    ON plants (health_status);

CREATE INDEX IF NOT EXISTS idx_plants_ndvi
    ON plants (ndvi);

CREATE INDEX IF NOT EXISTS idx_plants_plant_id
    ON plants (plant_id);

-- ── Helper view: summary per health status ─────────────────────────────────────
CREATE OR REPLACE VIEW plant_health_summary AS
SELECT
    health_status,
    health_color,
    COUNT(*)                           AS plant_count,
    ROUND(AVG(ndvi)::NUMERIC, 4)       AS avg_ndvi,
    ROUND(MIN(ndvi)::NUMERIC, 4)       AS min_ndvi,
    ROUND(MAX(ndvi)::NUMERIC, 4)       AS max_ndvi,
    ROUND(
        100.0 * COUNT(*) / SUM(COUNT(*)) OVER (),
        2
    )                                  AS pct_of_total
FROM plants
GROUP BY health_status, health_color
ORDER BY health_status;

COMMENT ON VIEW plant_health_summary IS
    'Quick summary of plant health distribution across all detected plants.';
