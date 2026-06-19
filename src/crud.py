from sqlalchemy.orm import Session
from sqlalchemy import func
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User, SalesTransaction
from .schemas import TransactionCreateSchema
from datetime import datetime

# --- User Authentication CRUD ---

def create_user(db: Session, username: str, password_raw: str) -> User:
    """Create a new user with a hashed password."""
    password_hash = generate_password_hash(password_raw)
    db_user = User(username=username, password_hash=password_hash)
    db.add(db_user)
    db.flush()  # Flushes to DB to populate ID and trigger constraints
    return db_user

def authenticate_user(db: Session, username: str, password_raw: str) -> User | None:
    """Validate username and password, returning User object if valid, else None."""
    db_user = db.query(User).filter(User.username == username).first()
    if not db_user:
        return None
    if check_password_hash(db_user.password_hash, password_raw):
        return db_user
    return None

# --- Sales Transaction CRUD ---

def create_transaction(db: Session, item: TransactionCreateSchema) -> SalesTransaction:
    """Add a single sales transaction to the database."""
    db_item = SalesTransaction(
        date=item.date,
        product_id=item.product_id,
        product_name=item.product_name,
        category=item.category,
        units_sold=item.units_sold,
        unit_price=item.unit_price,
        is_promo=item.is_promo
    )
    db.add(db_item)
    db.flush()
    return db_item

def bulk_create_transactions(db: Session, items: list[dict]) -> int:
    """
    Perform a fast bulk insert of transaction dictionaries.
    Returns the count of successfully inserted records.
    """
    if not items:
        return 0
    
    # Direct mapping using SQLAlchemy Core for high performance
    db.execute(
        SalesTransaction.__table__.insert(),
        items
    )
    return len(items)

def get_all_transactions(db: Session, start_date: str = None, end_date: str = None) -> list[SalesTransaction]:
    """Retrieve transactions filtered by start and end dates, ordered chronologically."""
    query = db.query(SalesTransaction)
    if start_date:
        query = query.filter(SalesTransaction.date >= start_date)
    if end_date:
        query = query.filter(SalesTransaction.date <= end_date)
    return query.order_by(SalesTransaction.date.asc(), SalesTransaction.product_id.asc()).all()

def get_dashboard_summary(db: Session) -> dict:
    """Generate high-level metrics for the analytics dashboard."""
    # Aggregates total units sold, total revenue, average units sold
    agg = db.query(
        func.sum(SalesTransaction.units_sold).label("total_units"),
        func.sum(SalesTransaction.units_sold * SalesTransaction.unit_price).label("total_revenue"),
        func.count(SalesTransaction.id).label("transaction_count"),
        func.min(SalesTransaction.date).label("min_date"),
        func.max(SalesTransaction.date).label("max_date")
    ).first()

    total_units = agg.total_units or 0
    total_revenue = agg.total_revenue or 0.0
    transaction_count = agg.transaction_count or 0
    aov = total_revenue / transaction_count if transaction_count > 0 else 0.0

    # Promo sales count
    promo_count = db.query(func.count(SalesTransaction.id)).filter(SalesTransaction.is_promo == True).scalar() or 0

    return {
        "total_units": int(total_units),
        "total_revenue": round(total_revenue, 2),
        "transaction_count": transaction_count,
        "average_order_value": round(aov, 2),
        "promo_transactions": promo_count,
        "date_range": f"{agg.min_date or 'N/A'} to {agg.max_date or 'N/A'}"
    }

def get_category_metrics(db: Session) -> list[dict]:
    """Calculate aggregate revenue and volume breakdown per product category."""
    results = db.query(
        SalesTransaction.category,
        func.sum(SalesTransaction.units_sold).label("units"),
        func.sum(SalesTransaction.units_sold * SalesTransaction.unit_price).label("revenue")
    ).group_by(SalesTransaction.category).all()

    return [
        {
            "category": r.category,
            "units": int(r.units or 0),
            "revenue": round(r.revenue or 0.0, 2)
        }
        for r in results
    ]
