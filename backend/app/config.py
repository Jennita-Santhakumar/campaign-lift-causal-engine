from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://causal:causal@localhost:5432/causal_impact"
    seed_n_customers: int = 4000
    true_ate: float = 12.50  # ground-truth per-period spend lift baked into the synthetic generator

    class Config:
        env_file = ".env"


settings = Settings()
