@echo off
cd /d "%~dp0"
echo === Step 1: Scrape (~45 pages for 300+ clean rows) ===
python scraper_zameen.py --pages 45 --output zameen_lahore_raw.csv
echo.
echo === Step 2: Clean ===
python preprocess.py --input zameen_lahore_raw.csv --output zameen_lahore_clean.csv --min-rows 300
echo.
echo === Step 3: Train ===
python train_ml.py --data zameen_lahore_clean.csv
echo.
echo === Done. Run: python predict.py ===
pause
