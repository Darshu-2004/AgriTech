"""
Single entry point for the plant-health index pipeline.

Run this and every stage (load plants -> sample NDVI -> sample NDRE -> save
CSV) executes automatically:

    python run.py
"""

from pipeline import run

if __name__ == "__main__":
    run()
