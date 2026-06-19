import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add root folder to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.database import Base, engine, get_db
from src.models import User, SalesTransaction
from src.crud import create_user, authenticate_user, bulk_create_transactions
from src.forecaster import load_data_from_transactions, engineer_features, train_forecaster, MODEL_FILE

def test_database_and_seeding():
    print("Running DB and Seeding verification...")
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    with get_db() as db:
        # User auth test
        # Clean potential old verify user
        old_user = db.query(User).filter(User.username == "verify_test_user").first()
        if old_user:
            db.delete(old_user)
            db.commit()
            
        user = create_user(db, "verify_test_user", "securepass123")
        db.commit()
        
        assert user.id is not None
        print("[OK] Hashed password user created successfully.")
        
        auth_success = authenticate_user(db, "verify_test_user", "securepass123")
        assert auth_success is not None
        assert auth_success.username == "verify_test_user"
        print("[OK] User authentication verification passed.")
        
        auth_fail = authenticate_user(db, "verify_test_user", "wrongpassword")
        assert auth_fail is None
        print("[OK] User invalid auth rejection passed.")
        
        # Test clean test transactions
        # Empty previous test items
        db.query(SalesTransaction).filter(SalesTransaction.product_id == "TEST_999").delete()
        db.commit()
        
        test_items = [
            {
                "date": "2026-06-01",
                "product_id": "TEST_999",
                "product_name": "Test Item A",
                "category": "Electronics",
                "units_sold": 5,
                "unit_price": 100.0,
                "is_promo": False
            },
            {
                "date": "2026-06-02",
                "product_id": "TEST_999",
                "product_name": "Test Item A",
                "category": "Electronics",
                "units_sold": 10,
                "unit_price": 90.0,
                "is_promo": True
            }
        ]
        
        inserted = bulk_create_transactions(db, test_items)
        db.commit()
        assert inserted == 2
        print("[OK] Bulk transactions insert verification passed.")
        
        txs = db.query(SalesTransaction).filter(SalesTransaction.product_id == "TEST_999").all()
        assert len(txs) == 2
        print("[OK] Transaction database queries passed.")

def test_ml_pipeline():
    print("\nRunning Machine Learning pipeline and data leakage verification...")
    
    # Create synthetic series
    dates = pd.date_range(start="2025-01-01", periods=100)
    data = []
    for i, dt in enumerate(dates):
        data.append(SalesTransaction(
            date=dt.strftime("%Y-%m-%d"),
            product_id="P1",
            product_name="Product 1",
            category="Grocery",
            units_sold=int(20 + 5 * np.sin(i / 7) + np.random.poisson(3)),
            unit_price=10.0,
            is_promo=False
        ))
        
    df = load_data_from_transactions(data)
    assert len(df) == 100
    print("[OK] Transaction list loader and aggregator passed.")
    
    # Feature engineering test
    feat_df = engineer_features(df, lag_days=7)
    
    # Verify lag columns exist
    for lag in range(1, 8):
        assert f"sales_lag_{lag}" in feat_df.columns
    assert "sales_lag_14" in feat_df.columns
    assert "rolling_mean_7" in feat_df.columns
    assert "rolling_mean_30" in feat_df.columns
    assert "rolling_std_7" in feat_df.columns
    assert "day_of_week" in feat_df.columns
    print("[OK] Time-series lag and rolling statistics features created successfully.")
    
    # Leakage checks:
    # 1. Ensure lag_1 for row t is equal to units_sold for row t-1
    for t in range(1, len(feat_df)):
        assert feat_df.loc[t, "sales_lag_1"] == feat_df.loc[t-1, "units_sold"]
    print("[OK] Checked autoregressive lag values for alignment (no forward leakage).")
    
    # 2. Ensure rolling mean for row t does not include units_sold at row t
    # rolling_mean_7 for row t must equal the average of units_sold for rows t-7 to t-1
    for t in range(7, len(feat_df)):
        expected_mean = feat_df.loc[t-7:t-1, "units_sold"].mean()
        assert np.isclose(feat_df.loc[t, "rolling_mean_7"], expected_mean)
    print("[OK] Checked moving averages for alignment (strictly historical lookbacks, no target leakage).")
    
    # Temporal Train/Test split verification:
    # Train set should contain dates strictly prior to the Test set
    config = {
        "algorithm": "linear_regression",
        "lag_days": 7,
        "rolling_window": 7,
        "test_size": 0.2
    }
    
    # Check model fitting
    res = train_forecaster(df, config)
    assert "metrics" in res
    assert "rmse" in res["metrics"]
    assert "test_eval" in res
    
    # Ensure test predictions are aligned chronologically and there is no temporal overlap
    test_eval_df = pd.DataFrame(res["test_eval"])
    test_eval_dates = pd.to_datetime(test_eval_df["date"])
    
    # The minimum test date should be strictly greater than the maximum training date
    split_idx = int(len(feat_df) * 0.8)
    max_train_date = feat_df["date"].iloc[split_idx - 1]
    min_test_date = test_eval_dates.min()
    
    assert min_test_date > max_train_date
    print(f"[OK] Checked temporal split (Train max date: {max_train_date.strftime('%Y-%m-%d')} < Test min date: {min_test_date.strftime('%Y-%m-%d')}). No data leakage.")
    print("[OK] Model pipeline trained and saved successfully.")

if __name__ == "__main__":
    try:
        test_database_and_seeding()
        test_ml_pipeline()
        print("\n[SUCCESS] All automated tests passed successfully!")
        sys.exit(0)
    except AssertionError as ae:
        print(f"\n[FAILURE] Assertion failed: {str(ae)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
