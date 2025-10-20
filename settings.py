# settings.py
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    ENVIRONMENT: str
    APP_NAME: str

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, value):
        allowed_environment = {"dev", "test", "prod"}
        if value not in allowed_environment:
            raise ValueError(
                f"Invalid environment: {value}. You should pick one of {allowed_environment}."
            )
        return value
