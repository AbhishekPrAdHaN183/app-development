import os
import sys
import random
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm

# Add pg5 root directory to python path if not already there
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import Base, engine, get_db
from src.models import User, SalesTransaction
from src.crud import create_user, bulk_create_transactions
from src.forecaster import load_data_from_transactions, train_forecaster

console = Console()

def init_db():
    """Create all database tables."""
    console.print("[yellow]Initializing database tables...[/yellow]")
    Base.metadata.create_all(bind=engine)
    console.print("[green]Database tables created successfully.[/green]\n")

def seed_data():
    """Seed the database with 2 years of realistic daily sales data with seasonality, trend, price elasticity, and promos."""
    init_db()
    
    with get_db() as db:
        # Check if data already exists
        count = db.query(SalesTransaction).count()
        if count > 0:
            confirm = Confirm.ask(f"Database already contains {count} transactions. Re-seed and append?")
            if not confirm:
                return
        
        console.print("[yellow]Generating 2 years of synthetic retail transactions...[/yellow]")
        
        # Define categories and base parameters
        categories = {
            "Electronics": {"base_qty": 12, "base_price": 299.99, "elasticity": -0.8},
            "Apparel": {"base_qty": 25, "base_price": 45.00, "elasticity": -1.2},
            "Grocery": {"base_qty": 60, "base_price": 8.50, "elasticity": -0.5},
            "Home & Garden": {"base_qty": 18, "base_price": 89.99, "elasticity": -1.0}
        }
        
        products = {
            "Electronics": [("E101", "Smart TV"), ("E102", "Bluetooth Speaker"), ("E103", "Wireless Headphoness")],
            "Apparel": [("A201", "Denim Jacket"), ("A202", "Running Shoes"), ("A203", "Cotton T-Shirt")],
            "Grocery": [("G301", "Organic Coffee"), ("G302", "Almond Milk"), ("G303", "Granola Bars")],
            "Home & Garden": [("H401", "LED Floor Lamp"), ("H402", "Ceramic Planter"), ("H403", "Ergonomic Pillow")]
        }

        start_date = datetime.now() - timedelta(days=730)  # 2 years ago
        end_date = datetime.now() - timedelta(days=1)      # Yesterday
        
        current_date = start_date
        records = []
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            day_of_week = current_date.weekday()
            month = current_date.month
            
            # Weekly seasonality: Sales peak on Friday (4), Saturday (5), and Sunday (6)
            weekly_mult = 1.0
            if day_of_week in [4, 5]:
                weekly_mult = 1.45
            elif day_of_week == 6:
                weekly_mult = 1.2
            
            # Yearly seasonality: peak in December (holiday) and July (summer sale)
            yearly_mult = 1.0
            if month == 12:
                yearly_mult = 1.5
            elif month == 7:
                yearly_mult = 1.25
            elif month in [1, 2]:
                yearly_mult = 0.85
                
            # Upward overall trend (+10% growth per year)
            years_passed = (current_date - start_date).days / 365.25
            trend_mult = 1.0 + (0.10 * years_passed)
            
            for category, info in categories.items():
                # Loop through products in category
                prod_list = products[category]
                for prod_id, prod_name in prod_list:
                    # Randomize promo occurrence (15% chance per day)
                    is_promo = random.random() < 0.15
                    
                    # Determine unit price (promo discount is 15-25% off)
                    base_price = info["base_price"]
                    if is_promo:
                        discount = random.uniform(0.15, 0.25)
                        unit_price = base_price * (1.0 - discount)
                    else:
                        # Slight normal price fluctuations
                        unit_price = base_price * random.uniform(0.97, 1.03)
                    
                    # Price elasticity effect on quantity
                    price_ratio = unit_price / base_price
                    elasticity_mult = price_ratio ** info["elasticity"]
                    
                    # Promo volume boost (+50% to +100%)
                    promo_mult = random.uniform(1.5, 2.0) if is_promo else 1.0
                    
                    # Base target sales quantity
                    base_qty = info["base_qty"] / len(prod_list)
                    
                    # Compute expected units sold
                    expected_qty = base_qty * weekly_mult * yearly_mult * trend_mult * elasticity_mult * promo_mult
                    
                    # Add noise (Poisson distribution or normal with clipping)
                    units_sold = int(max(0, np.random.normal(expected_qty, np.sqrt(expected_qty) + 1.0)))
                    
                    # Only insert transactions where units_sold > 0 (realistic)
                    if units_sold > 0:
                        records.append({
                            "date": date_str,
                            "product_id": prod_id,
                            "product_name": prod_name,
                            "category": category,
                            "units_sold": units_sold,
                            "unit_price": round(unit_price, 2),
                            "is_promo": is_promo
                        })
            
            # Flush batch to DB occasionally to manage memory
            if len(records) >= 2000:
                bulk_create_transactions(db, records)
                records = []
                
            current_date += timedelta(days=1)
            
        # Insert remaining records
        if records:
            bulk_create_transactions(db, records)
            
        total_tx = db.query(SalesTransaction).count()
        console.print(f"[green]Data seeding finished successfully! Added database transactions. Total records in database: {total_tx}[/green]\n")

def run_train_pipeline():
    """Load transaction data, trigger model training, and display metrics in CLI."""
    init_db()
    with get_db() as db:
        transactions = db.query(SalesTransaction).all()
        if not transactions:
            console.print("[red]Error: Database is empty. Please run Data Seeding first.[/red]\n")
            return
        
        # Load and aggregate data
        daily_df = load_data_from_transactions(transactions)
        
        console.print("[yellow]Training machine learning forecasting models...[/yellow]")
        
        # Configure model parameters
        config = {
            "algorithm": "random_forest",
            "lag_days": 7,
            "rolling_window": 7,
            "test_size": 0.2
        }
        
        try:
            res = train_forecaster(daily_df, config)
            
            # Display results panel
            metrics = res["metrics"]
            importance_rows = ""
            for item in res["feature_importance"]:
                importance_rows += f" • {item['feature']}: {item['importance']:.3f}\n"
                
            panel_content = (
                f"[bold cyan]Model Algorithm:[/bold cyan] {res['algorithm'].upper()}\n"
                f"[bold cyan]Root Mean Squared Error (RMSE):[/bold cyan] {metrics['rmse']:.2f} units\n"
                f"[bold cyan]Mean Absolute Error (MAE):[/bold cyan] {metrics['mae']:.2f} units\n"
                f"[bold cyan]Coefficient of Determination (R²):[/bold cyan] {metrics['r2']:.4f}\n\n"
                f"[bold green]Top Feature Importances:[/bold green]\n{importance_rows}"
            )
            console.print(Panel(panel_content, title="Model Evaluation Results (Chronological Split Test)", border_style="green"))
            
        except Exception as e:
            console.print(f"[red]Error training model: {str(e)}[/red]\n")

def create_admin():
    """Create a default administrator login."""
    init_db()
    username = Prompt.ask("Enter admin username", default="admin")
    password = Prompt.ask("Enter admin password", password=True)
    
    with get_db() as db:
        # Check if user already exists
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            confirm = Confirm.ask(f"User '{username}' already exists. Overwrite password?")
            if confirm:
                db.delete(existing)
                db.flush()
            else:
                return
        
        try:
            create_user(db, username, password)
            console.print(f"[green]Admin user '{username}' created successfully.[/green]\n")
        except Exception as e:
            console.print(f"[red]Error creating user: {str(e)}[/red]\n")

def start_server():
    """Launch the Flask web server."""
    init_db()
    # Ensure there is an admin user created, otherwise create a default one
    with get_db() as db:
        if db.query(User).count() == 0:
            console.print("[yellow]No users found in database. Creating default admin user (admin/admin)...[/yellow]")
            create_user(db, "admin", "admin")
            db.commit()
            
        if db.query(SalesTransaction).count() == 0:
            console.print("[yellow]No transactions found in database. Proactively seeding initial data...[/yellow]")
            seed_data()
            
        # Check if model exists, if not, train it
        from src.forecaster import MODEL_FILE
        if not os.path.exists(MODEL_FILE):
            console.print("[yellow]No trained ML model found. Training default model...[/yellow]")
            run_train_pipeline()

    console.print("[green]Launching web application local development server...[/green]")
    # Import app inside to avoid circular dependencies
    from src.app import app
    app.run(host="127.0.0.1", port=8000, debug=True)

def show_menu():
    """Display CLI Menu options."""
    title = Text("Nexus Demand Forecaster - Management Console", style="bold magenta")
    menu_panel = Panel(
        "1. [bold green]Seed Historical Sales Data[/bold green] (Generate 2 years daily logs)\n"
        "2. [bold cyan]Train Forecasting Models[/bold cyan] (Fit and evaluate ML pipeline)\n"
        "3. [bold yellow]Create Admin Credentials[/bold yellow] (Authentication helper)\n"
        "4. [bold blue]Start Flask Web Server[/bold blue] (Launch dashboard at port 8000)\n"
        "5. [bold red]Exit[/bold red]",
        title=title,
        border_style="magenta"
    )
    console.print(menu_panel)

def main():
    # Handle arguments if any, otherwise interactive CLI
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "seed":
            seed_data()
        elif arg == "train":
            run_train_pipeline()
        elif arg == "admin":
            create_admin()
        elif arg == "server":
            start_server()
        else:
            console.print(f"[red]Unknown argument: {arg}[/red]")
        sys.exit(0)

    while True:
        show_menu()
        choice = Prompt.ask("Select an option", choices=["1", "2", "3", "4", "5"])
        if choice == "1":
            seed_data()
        elif choice == "2":
            run_train_pipeline()
        elif choice == "3":
            create_admin()
        elif choice == "4":
            try:
                start_server()
            except KeyboardInterrupt:
                console.print("\n[yellow]Server stopped.[/yellow]\n")
        elif choice == "5":
            console.print("[yellow]Exiting Nexus Demand Forecaster CLI. Goodbye![/yellow]")
            break

if __name__ == "__main__":
    main()
