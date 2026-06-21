-- ============================================================
-- schema.sql
-- PostGIS schema for pineapple plant health data
-- Run once to set up the database.
-- ============================================================

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- ── Main pineapple crops table ──────────────────────────────
CREATE TABLE IF NOT EXISTS pineapple_crops (
    plant_id                    VARCHAR(100) PRIMARY KEY,
    sector_id                   INT,
    sector_label                VARCHAR(50),
    pixel_x                     INT,
    pixel_y                     INT,
    geo_x                       DOUBLE PRECISION,
    geo_y                       DOUBLE PRECISION,
    geom                        GEOMETRY(Point, 32651),   -- PostGIS spatial point index (SRID 32651)
    
    -- Biometrics & Stage
    area_px                     INT,
    canopy_area                 DOUBLE PRECISION,         -- Physical footprint area in m2
    predicted_growth_stage      VARCHAR(50) CHECK (
        predicted_growth_stage IN ('Seedling', 'Vegetative', 'Flowering', 'Fruiting', 'Mature')
    ),
    
    -- Health scoring
    health_score                DOUBLE PRECISION,         -- 0 to 100 continuous score
    health_status               VARCHAR(50) CHECK (
        health_status IN ('Healthy', 'Moderate', 'Stressed', 'Critical', 'BoundaryLimit')
    ),
    health_status_code          INT CHECK (
        health_status_code IN (0, 1, 2, 3, 4)
    ),
    health_color                VARCHAR(20),
    
    -- Raw indices
    osavi                       DOUBLE PRECISION,
    ndvi                        DOUBLE PRECISION,
    ndre                        DOUBLE PRECISION,
    
    -- Flight-specific telemetry
    flight_date                 TIMESTAMP,
    slope_degrees               DOUBLE PRECISION,
    drainage_accumulation       DOUBLE PRECISION,
    estimated_height            DOUBLE PRECISION,
    canopy_circularity          DOUBLE PRECISION,
    nearest_neighbor_dist_m     DOUBLE PRECISION,
    
    -- Calculated temporal changes
    delta_canopy_area           DOUBLE PRECISION,
    delta_ndvi                  DOUBLE PRECISION,
    delta_ndre                  DOUBLE PRECISION,
    stagnation_flag             BOOLEAN,
    regression_flag             BOOLEAN,
    
    -- Audit
    created_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Spatial index for high-speed maps / spatial queries
CREATE INDEX IF NOT EXISTS idx_pineapple_crops_geom
    ON pineapple_crops USING GIST (geom);

-- Indexes for fast query lookups on dashboard filter keys
CREATE INDEX IF NOT EXISTS idx_pineapple_crops_health_status
    ON pineapple_crops (health_status);

CREATE INDEX IF NOT EXISTS idx_pineapple_crops_stage
    ON pineapple_crops (predicted_growth_stage);

CREATE INDEX IF NOT EXISTS idx_pineapple_crops_flight_date
    ON pineapple_crops (flight_date);