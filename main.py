import argparse
from dotenv import load_dotenv
from settings import Settings
from pathlib import Path

ENVS = {"dev", "test", "prod"}


def export_envs(environment: str = "dev") -> None:
    if environment not in ENVS:
        raise ValueError("Error, pick one of dev/test/prod")
    dotenv_path = Path("config") / f".env.{environment}"
    if not dotenv_path.exists():
        raise FileNotFoundError("File not Found")
    load_dotenv(dotenv_path=dotenv_path, override=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load environment variables from specified.env file."
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="dev",
        help="The environment to load (dev, test, prod)",
    )
    args = parser.parse_args()

    export_envs(args.environment)
    settings = Settings()

    print("APP_NAME: ", settings.APP_NAME)
    print("ENVIRONMENT: ", settings.ENVIRONMENT)
