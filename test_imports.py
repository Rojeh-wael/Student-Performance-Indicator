import sys
import traceback
try:
    from flask import Flask
    print("[OK] Flask imported")
    from src.pipeline.predict_pipeline import PredictPipeline, CustomData
    print("[OK] predict_pipeline imported")
    from application import app
    print("[OK] application imported")
    print("\n✓ All imports successful!")
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
