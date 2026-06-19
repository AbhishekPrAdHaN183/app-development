import os
import csv
import io
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from functools import wraps

from .database import get_db
from .crud import authenticate_user, create_transaction, bulk_create_transactions, get_all_transactions, get_dashboard_summary, get_category_metrics
from .schemas import UserLoginSchema, TransactionCreateSchema, ModelTrainConfigSchema
from .forecaster import load_data_from_transactions, train_forecaster, forecast_future_demand, MODEL_FILE, get_feature_importance
import joblib

# Create flask app, configuring it to look at 'static' directory for templates (index.html)
# and for static assets (styles.css, app.js)
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))

app = Flask(
    __name__,
    template_folder=STATIC_DIR,
    static_folder=STATIC_DIR
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "nexus-demand-forecaster-secret-key-1938")

# --- Authentication Helpers ---

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session:
            # If it's an API request, return 401 JSON, else redirect to home (which shows login)
            if request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized. Please log in."}), 401
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated_function

# --- Page Routes ---

@app.route("/", methods=["GET", "POST"])
def home():
    """Render dashboard page or handle login submission depending on session state."""
    if "username" in session:
        return render_template("index.html")
    
    # Handle GET or POST for login
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        try:
            # Pydantic validation
            UserLoginSchema(username=username, password=password)
            
            with get_db() as db:
                user = authenticate_user(db, username, password)
                if user:
                    session["username"] = user.username
                    return redirect(url_for("home"))
                else:
                    flash("Invalid username or password.", "error")
        except ValidationError as ve:
            errors = ", ".join([f"{err['loc'][0]}: {err['msg']}" for err in ve.errors()])
            flash(f"Validation Error: {errors}", "error")
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def api_login():
    """API-friendly login endpoint returning JSON."""
    data = request.get_json() or {}
    try:
        schema = UserLoginSchema(**data)
        with get_db() as db:
            user = authenticate_user(db, schema.username, schema.password)
            if user:
                session["username"] = user.username
                return jsonify({"status": "success", "username": user.username})
            return jsonify({"error": "Invalid username or password."}), 401
    except ValidationError as ve:
        return jsonify({"error": ve.errors()}), 400

@app.route("/logout")
def logout():
    """Clear session data and redirect to login screen."""
    session.pop("username", None)
    return redirect(url_for("home"))

# --- Analytics & ML API Endpoints ---

@app.route("/api/stats", methods=["GET"])
@login_required
def api_stats():
    """Fetch high-level inventory/sales stats and current model training status."""
    try:
        with get_db() as db:
            summary = get_dashboard_summary(db)
            categories = get_category_metrics(db)
            
        # Check if model exists and extract info
        model_status = {"trained": False}
        if os.path.exists(MODEL_FILE):
            pipeline = joblib.load(MODEL_FILE)
            model_status = {
                "trained": True,
                "algorithm": pipeline.get("algorithm", "Unknown"),
                "lag_days": pipeline.get("lag_days", 7),
                "metrics": pipeline.get("metrics", {}),
                "feature_importance": get_feature_importance(pipeline["model"], pipeline["feature_cols"])
            }
            
        return jsonify({
            "summary": summary,
            "categories": categories,
            "model_status": model_status
        })
    except SQLAlchemyError as se:
        return jsonify({"error": f"Database error: {str(se)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/historical-data", methods=["GET"])
@login_required
def api_historical_data():
    """Get aggregated daily historical sales data for charting."""
    try:
        with get_db() as db:
            transactions = get_all_transactions(db)
            if not transactions:
                return jsonify([])
            df = load_data_from_transactions(transactions)
            
        # Format dates back to string for JSON serialization
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/forecast", methods=["GET"])
@login_required
def api_forecast():
    """Generate recursive multi-step forecasting for the next N days (default 30)."""
    steps = request.args.get("steps", default=30, type=int)
    try:
        with get_db() as db:
            transactions = get_all_transactions(db)
            if not transactions:
                return jsonify({"error": "No historical transactions found. Please seed data first."}), 400
            df = load_data_from_transactions(transactions)
        
        # Check if model exists
        if not os.path.exists(MODEL_FILE):
            return jsonify({"error": "ML Forecaster has not been trained yet. Click the Retrain button."}), 400
            
        predictions = forecast_future_demand(df, steps=steps)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict-elasticity", methods=["POST"])
@login_required
def api_predict_elasticity():
    """
    Perform a single day prediction simulating price elasticity and promo uplift.
    Uses current baseline lags to isolate elasticity impact.
    """
    data = request.get_json() or {}
    price = data.get("unit_price")
    is_promo = data.get("is_promo", False)
    
    if price is None:
        return jsonify({"error": "Missing unit_price"}), 400
        
    try:
        price = float(price)
        if price <= 0:
            return jsonify({"error": "Price must be positive"}), 400
            
        # Get historical average lags as a baseline
        with get_db() as db:
            transactions = get_all_transactions(db)
            if not transactions:
                return jsonify({"error": "No historical data to establish baseline lags."}), 400
            df = load_data_from_transactions(transactions)
        
        # Check model file
        if not os.path.exists(MODEL_FILE):
            return jsonify({"error": "Model is not trained."}), 400
            
        import pandas as pd  # Ensure pandas is imported if not globally available
        pipeline = joblib.load(MODEL_FILE)
        model = pipeline["model"]
        scaler = pipeline["scaler"]
        feature_cols = pipeline["feature_cols"]
        
        # Build features using historical averages
        feat_dict = {}
        for col in feature_cols:
            if col == "unit_price":
                feat_dict[col] = price
            elif col == "is_promo":
                feat_dict[col] = 1 if is_promo else 0
            elif "lag" in col or "rolling" in col:
                # Use mean of recent sales
                feat_dict[col] = float(df["units_sold"].tail(30).mean())
            elif col == "day_of_week":
                feat_dict[col] = 2  # Wednesday (neutral weekday)
            elif col == "month":
                feat_dict[col] = 6  # June (neutral month)
            elif col == "day_of_year":
                feat_dict[col] = 170
            elif col == "is_weekend":
                feat_dict[col] = 0
            else:
                feat_dict[col] = 0.0
                
        X_row = pd.DataFrame([feat_dict])[feature_cols]
        X_scaled = scaler.transform(X_row)
        
        pred_units = model.predict(X_scaled)[0]
        pred_units = max(0, float(pred_units))
        
        return jsonify({
            "unit_price": price,
            "is_promo": is_promo,
            "predicted_units_sold": round(pred_units, 2),
            "estimated_revenue": round(pred_units * price, 2)
        })
    except ValueError:
        return jsonify({"error": "Invalid pricing float input"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/train", methods=["POST"])
@login_required
def api_train():
    """Trigger manual retraining of the forecasting models with custom settings."""
    data = request.get_json() or {}
    try:
        config_schema = ModelTrainConfigSchema(**data)
        config = config_schema.model_dump()
        
        with get_db() as db:
            transactions = get_all_transactions(db)
            if not transactions:
                return jsonify({"error": "No transaction records available. Seed data first."}), 400
            daily_df = load_data_from_transactions(transactions)
        
        # Train forecaster
        results = train_forecaster(daily_df, config)
        return jsonify({
            "status": "success",
            "algorithm": results["algorithm"],
            "metrics": results["metrics"],
            "test_eval": results["test_eval"],
            "feature_importance": results["feature_importance"]
        })
    except ValidationError as ve:
        return jsonify({"error": ve.errors()}), 400
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal training failure: {str(e)}"}), 500

# --- Bulk Operations / Data Upload ---

@app.route("/api/add-transaction", methods=["POST"])
@login_required
def api_add_transaction():
    """Manually add a single transaction to the database."""
    data = request.json or {}
    try:
        tx_schema = TransactionCreateSchema(**data)
        with get_db() as db:
            tx = create_transaction(db, tx_schema)
            db.commit()
            return jsonify({
                "status": "success",
                "id": tx.id,
                "message": f"Added transaction for {tx.product_name}."
            })
    except ValidationError as ve:
        return jsonify({"error": ve.errors()}), 400
    except SQLAlchemyError as se:
        return jsonify({"error": f"Database insertion error: {str(se)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/upload", methods=["POST"])
@login_required
def api_upload():
    """Upload a CSV file containing transaction batches."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
        
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "File must be in CSV format (.csv)"}), 400
        
    try:
        # Read file streams
        stream = io.StringIO(file.stream.read().decode("utf-8"), newline=None)
        reader = csv.DictReader(stream)
        
        # Validate headers
        required_headers = {"date", "product_id", "product_name", "category", "units_sold", "unit_price", "is_promo"}
        missing_headers = required_headers - set(reader.fieldnames or [])
        if missing_headers:
            return jsonify({"error": f"Missing required headers: {list(missing_headers)}"}), 400
            
        transactions_to_insert = []
        row_count = 0
        
        for row in reader:
            row_count += 1
            # Parse values for Pydantic
            try:
                is_promo_parsed = row.get("is_promo", "false").lower() in ("true", "1", "yes")
                parsed_row = {
                    "date": row.get("date", "").strip(),
                    "product_id": row.get("product_id", "").strip(),
                    "product_name": row.get("product_name", "").strip(),
                    "category": row.get("category", "").strip(),
                    "units_sold": int(row.get("units_sold", "0")),
                    "unit_price": float(row.get("unit_price", "0")),
                    "is_promo": is_promo_parsed
                }
                # Pydantic schema validation
                TransactionCreateSchema(**parsed_row)
                transactions_to_insert.append(parsed_row)
            except (ValueError, ValidationError) as err:
                return jsonify({"error": f"CSV Row {row_count} parsing validation failed: {str(err)}"}), 400
                
        # Commit to DB
        with get_db() as db:
            inserted = bulk_create_transactions(db, transactions_to_insert)
            db.commit()
            
        return jsonify({
            "status": "success",
            "inserted_count": inserted,
            "message": f"Successfully parsed and saved {inserted} transaction records."
        })
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV file: {str(e)}"}), 500

# --- Error Handlers ---

@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Resource not found"}), 404
    return render_template("index.html"), 404

@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error"}), 500
    return render_template("index.html"), 500
