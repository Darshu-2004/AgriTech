import os
import pandas as pd
import psycopg2
import io

# Database connection configuration (standard PostgreSQL env variables)
DB_HOST = os.getenv("PGHOST", "localhost")
DB_PORT = os.getenv("PGPORT", "5432")
DB_DATABASE = os.getenv("PGDATABASE", "plantation_db")
DB_USER = os.getenv("PGUSER", "postgres")
DB_PASSWORD = os.getenv("PGPASSWORD", "postgres")
DB_SRID = os.getenv("PGSRID", "32651")  # Default UTM projection zone (e.g., Zone 51S)

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.dirname(script_dir)
csv_path = os.path.join(workspace_dir, "plants_with_predictions_health.csv")

def export_to_db():
    print("=== PIPELINE STEP 4: POSTGRESQL/POSTGIS EXPORTER ===")
    
    if not os.path.exists(csv_path):
        print(f"Error: Final output CSV not found at: {csv_path}. Please run the pipeline first.")
        return False
        
    print(f"Connecting to database '{DB_DATABASE}' on {DB_HOST}:{DB_PORT}...")
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_DATABASE,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=3
        )
        cursor = conn.cursor()
    except Exception as e:
        print(f"Connection Failed: {e}")
        print("Note: If you do not have a live database running, this is expected. You can run")
        print("      PostgreSQL locally or set connection details using environment variables.")
        return False

    try:
        # 1. Enable PostGIS Extension
        print("Enabling PostGIS extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
        
        # 2. Create Table Schema
        print("Refreshing table schema for 'pineapple_crops'...")
        cursor.execute("DROP TABLE IF EXISTS pineapple_crops;")
        
        create_table_query = f"""
        CREATE TABLE pineapple_crops (
            plant_id VARCHAR(100) PRIMARY KEY,
            instance_id INT,
            sector_id INT,
            sector_label VARCHAR(50),
            row_index INT,
            col_index INT,
            pixel_x INT,
            pixel_y INT,
            geo_x DOUBLE PRECISION,
            geo_y DOUBLE PRECISION,
            geom GEOMETRY(Point, {DB_SRID}),
            area_px INT,
            canopy_area_m2 DOUBLE PRECISION,
            bbox_x INT,
            bbox_y INT,
            bbox_width INT,
            bbox_height INT,
            mask_file VARCHAR(255),
            predicted_growth_stage_name VARCHAR(50),
            health_score DOUBLE PRECISION,
            health_status VARCHAR(50),
            health_color VARCHAR(20),
            osavi DOUBLE PRECISION,
            ndvi DOUBLE PRECISION,
            ndre DOUBLE PRECISION,
            osavi_smoothed DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)
        
        # 3. Read CSV and prepare for bulk COPY
        print("Formatting dataset for bulk copy...")
        df = pd.read_csv(csv_path)
        # Ensure NaNs are mapped to None (becomes SQL NULL)
        df = df.where(pd.notnull(df), None)
        
        # 4. Perform high-speed bulk COPY from StringIO buffer
        print("Bulk uploading records to PostgreSQL (COPY EXPERT)...")
        buffer = io.StringIO()
        columns = [
            'plant_id', 'instance_id', 'sector_id', 'sector_label', 'row_index', 'col_index',
            'pixel_x', 'pixel_y', 'geo_x', 'geo_y', 'area_px', 'canopy_area_m2',
            'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'mask_file',
            'predicted_growth_stage_name', 'health_score', 'health_status', 'health_color',
            'osavi', 'ndvi', 'ndre', 'osavi_smoothed'
        ]
        
        df[columns].to_csv(buffer, sep='\t', header=False, index=False, na_rep='\\N')
        buffer.seek(0)
        
        copy_query = f"""
        COPY pineapple_crops ({', '.join(columns)}) 
        FROM STDIN WITH DELIMITER '\t' NULL '\\N';
        """
        cursor.copy_expert(copy_query, buffer)
        
        # 5. Populate geometry column
        print(f"Populating PostGIS Geometry column (Point, SRID {DB_SRID})...")
        update_geom_query = f"""
        UPDATE pineapple_crops 
        SET geom = ST_SetSRID(ST_MakePoint(geo_x, geo_y), {DB_SRID});
        """
        cursor.execute(update_geom_query)
        
        # 6. Create GiST spatial index for high-speed spatial querying
        print("Creating GiST spatial index on geometries...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pineapple_crops_geom ON pineapple_crops USING gist(geom);")
        
        conn.commit()
        print(f"Success: Exported {len(df)} plant records to PostgreSQL/PostGIS database table 'pineapple_crops'!")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"Error during export: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    export_to_db()
