# 🌿 NDVI Plant Health Pipeline

Extract NDVI values from a drone-captured visual NDVI raster for each detected plant, classify plant health, and load everything into PostgreSQL + PostGIS.

---

## Project structure

```
ndvi-pipeline/
├── scripts/
│   ├── extract_ndvi.py      # Step 1: NDVI extraction & classification
│   └── load_to_postgis.py   # Step 2: Load results into PostGIS
├── sql/
│   └── schema.sql           # Database schema (run once)
├── data/                    # ← create this folder, put your input files here (gitignored)
│   ├── plants.geojson
│   ├── plant_health.tif
│   └── orthomosaic.tif
├── output/                  # ← create this folder, results land here (gitignored)
│   ├── plants_ndvi.json
│   └── preview.png
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Create folders and place input files

```bash
mkdir data
mkdir output
```

Drop your 3 input files into the `data/` folder:
- `plants.geojson` — detected plant locations (UTM coordinates)
- `plant_health.tif` — visual NDVI raster (green = healthy, red = diseased)
- `orthomosaic.tif` — true colour orthomosaic (used only for preview background)

### 2. Install dependencies

```bash
pip install rasterio numpy matplotlib pyproj psycopg2-binary tqdm
```

### 3. Extract NDVI values

**Mac/Linux:**
```bash
python scripts/extract_ndvi.py \
  --plants  data/plants.geojson \
  --ndvi    data/plant_health.tif \
  --ortho   data/orthomosaic.tif \
  --output  output/plants_ndvi.json \
  --preview output/preview.png
```

**Windows (PowerShell):**
```powershell
python scripts/extract_ndvi.py --plants data/plants.geojson --ndvi data/plant_health.tif --ortho data/orthomosaic.tif --output output/plants_ndvi.json --preview output/preview.png
```

This produces:
- `output/plants_ndvi.json` — GeoJSON with `ndvi`, `health_status`, `health_color` added to each plant
- `output/preview.png` — orthomosaic with coloured dots overlaid for visual verification

### 4. Set up the database (Docker)

Install Docker Desktop from https://www.docker.com/products/docker-desktop/

Start the PostGIS container:
```bash
docker run -d --name ndvi-db \
  -e POSTGRES_USER=ndvi \
  -e POSTGRES_PASSWORD=ndvi123 \
  -e POSTGRES_DB=ndvidb \
  -p 5433:5432 \
  postgis/postgis:15-3.3
```

> ⚠️ We use port **5433** (not 5432) to avoid conflicts with any local PostgreSQL already installed on your machine.

Wait 15 seconds for the DB to initialize, then create the schema:

**Mac/Linux:**
```bash
cat sql/schema.sql | docker exec -i ndvi-db psql -U ndvi -d ndvidb
```

**Windows (PowerShell):**
```powershell
Get-Content sql/schema.sql | docker exec -i ndvi-db psql -U ndvi -d ndvidb
```

You should see `CREATE TABLE`, `CREATE INDEX` printed out.

### 5. Load into PostGIS

```bash
python scripts/load_to_postgis.py \
  --input output/plants_ndvi.json \
  --dsn "postgresql://ndvi:ndvi123@127.0.0.1:5433/ndvidb"
```

> ⚠️ Use `127.0.0.1` instead of `localhost` on Windows to ensure the connection reaches Docker correctly.

Add `--truncate` to clear the table before inserting (useful for re-runs).

### 6. Verify in Docker Desktop

Open Docker Desktop → click `ndvi-db` → click the **Exec** tab → run:

```bash
psql -U ndvi -d ndvidb
```

Then:
```sql
SELECT * FROM plant_health_summary;
```

Expected output:
```
 health_status | health_color | plant_count | avg_ndvi | min_ndvi | max_ndvi | pct_of_total
---------------+--------------+-------------+----------+----------+----------+--------------
 diseased      | #E74C3C      |         934 |  -0.0052 |  -0.7131 |   0.0995 |        17.93
 healthy       | #2ECC71      |        3083 |   0.7171 |   0.4016 |   0.8676 |        59.20
 moderate      | #F39C12      |        1191 |   0.2409 |   0.1024 |   0.3984 |        22.87
```

To exit psql: `\q`

---

## Restarting the database

When you restart your PC, Docker containers stop. To bring the DB back up:

```bash
docker start ndvi-db
```

Your data will still be there.

---

## Cloud deployment

On a cloud server (AWS, GCP, Azure, DigitalOcean), install Docker Engine and run:

```bash
curl -fsSL https://get.docker.com | sh

docker run -d --name ndvi-db \
  --restart always \
  -e POSTGRES_USER=ndvi \
  -e POSTGRES_PASSWORD=ndvi123 \
  -e POSTGRES_DB=ndvidb \
  -v ndvi-data:/var/lib/postgresql/data \
  -p 5433:5432 \
  postgis/postgis:15-3.3
```

The `--restart always` flag keeps the DB running across reboots.
The `-v ndvi-data:...` flag persists data even if the container is removed.

For managed cloud databases (AWS RDS, GCP Cloud SQL, Azure Database), just point the `--dsn` flag at their connection string — no Docker needed on those platforms.

---

## NDVI thresholds & colours

| Status   | NDVI range  | Hex colour   |
|----------|-------------|--------------|
| Healthy  | ≥ 0.40      | `#2ECC71` 🟢 |
| Moderate | 0.10 – 0.40 | `#F39C12` 🟡 |
| Diseased | < 0.10      | `#E74C3C` 🔴 |

Override defaults at extraction time:

```bash
python scripts/extract_ndvi.py ... --healthy 0.35 --moderate 0.15
```

### How NDVI is computed

The visual NDVI raster (`plant_health.tif`) encodes health as a colour gradient — green for healthy, red for diseased. The pipeline:

1. Takes each plant's UTM coordinate from the GeoJSON
2. Converts it to a pixel row/col in `plant_health.tif` using rasterio
3. Reads the Red and Green band values at that pixel
4. Computes: `NDVI_proxy = (Green − Red) / (Green + Red)`

Result is in the range `[-1, +1]`. The orthomosaic is only used for the preview background — it has no effect on NDVI values.

---

## Output JSON schema

Each feature in `plants_ndvi.json` contains the original plant properties **plus**:

```json
{
  "ndvi": 0.705,
  "health_status": "healthy",
  "health_color": "#2ECC71",
  "raster_crs": "EPSG:32647"
}
```

Coordinates are stored in UTM (EPSG:32647) in the JSON and automatically reprojected to WGS84 (EPSG:4326) when loaded into PostGIS.

---

## Useful PostGIS queries

```sql
-- Overall health distribution
SELECT * FROM plant_health_summary;

-- Individual plant records
SELECT plant_id, ndvi, health_status, health_color FROM plants LIMIT 10;

-- All diseased plants
SELECT plant_id, ndvi FROM plants WHERE health_status = 'diseased' ORDER BY ndvi ASC;

-- Count by sector and health
SELECT sector_label, health_status, COUNT(*) AS n
FROM plants
GROUP BY sector_label, health_status
ORDER BY sector_label, health_status;

-- Average NDVI per row
SELECT row_index, ROUND(AVG(ndvi)::NUMERIC, 4) AS avg_ndvi
FROM plants
GROUP BY row_index
ORDER BY row_index;

-- Diseased plants within a bounding box
SELECT plant_id, ndvi, ST_AsGeoJSON(geom)
FROM plants
WHERE health_status = 'diseased'
  AND ST_Within(geom, ST_MakeEnvelope(100.5, 13.0, 101.0, 13.5, 4326));
```

---

## Notes

- Tested on 5,208 plants with a 4,359 × 6,728 px raster — runs in under 1 second for extraction.
- Both scripts are idempotent: re-running `extract_ndvi.py` overwrites the JSON, and `load_to_postgis.py --truncate` clears the table before re-inserting.
- On Windows, always use `127.0.0.1` instead of `localhost` in the DSN, and port `5433` if you have a local PostgreSQL installed.
- The `plant_health_summary` view is created automatically by the schema and gives an instant breakdown of health distribution.

## Future improvements

- Plant centre coordinate to polygon coordinates so that we can add crown size as well and mean ndvi , ndre and osavi will be more accurate
- NDRE and OSAVI addition

