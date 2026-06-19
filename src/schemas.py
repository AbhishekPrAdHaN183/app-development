from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime

class UserLoginSchema(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4, max_length=100)

class TransactionCreateSchema(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")  # YYYY-MM-DD
    product_id: str = Field(..., min_length=1)
    product_name: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    units_sold: int = Field(..., ge=0)
    unit_price: float = Field(..., gt=0.0)
    is_promo: bool = Field(default=False)

    @field_validator("date")
    @classmethod
    def validate_date_range(cls, v: str) -> str:
        try:
            parsed_date = datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Date must be a valid calendar date in YYYY-MM-DD format")
        
        # Prevent extreme historical/future dates
        if parsed_date.year < 2015 or parsed_date.year > 2035:
            raise ValueError("Date year must be between 2015 and 2035")
        return v

class ModelTrainConfigSchema(BaseModel):
    algorithm: str = Field("random_forest", description="Choose between 'linear_regression' or 'random_forest'")
    lag_days: int = Field(7, ge=1, le=30, description="Number of past lag days to use as features")
    rolling_window: int = Field(7, ge=2, le=30, description="Window size for rolling statistics")
    test_size: float = Field(0.2, gt=0.0, lt=0.5, description="Proportion of temporal test set")

    @field_validator("algorithm")
    @classmethod
    def validate_algo(cls, v: str) -> str:
        valid_algos = ["linear_regression", "random_forest"]
        if v.lower() not in valid_algos:
            raise ValueError(f"Algorithm must be one of {valid_algos}")
        return v.lower()
