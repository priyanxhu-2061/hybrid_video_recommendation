from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379/0"

    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    ARTIFACT_DIR: str = "../recsys/artifacts"
    MODEL_VERSION: str = "latest"

    # Serving knobs - tune these without retraining.
    CANDIDATES_PER_SOURCE: int = 200
    FINAL_TOP_K: int = 20
    RECOMMENDATION_CACHE_TTL: int = 300
    COLD_START_THRESHOLD: int = 5

    CORS_ORIGINS: list[str] = ["http://localhost:5173"]


settings = Settings()
