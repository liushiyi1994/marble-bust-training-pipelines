import os
from collections.abc import Mapping


def required_env_vars(scope: str) -> list[str]:
    training = ["HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    if scope == "training":
        return training
    if scope == "runpod":
        return training + ["RUNPOD_API_KEY"]
    raise ValueError(f"unknown scope {scope}")


def validate_env(scope: str, env: Mapping[str, str] | None = None) -> None:
    env_vars = os.environ if env is None else env
    required = required_env_vars(scope)
    missing = [name for name in required if name not in env_vars]
    if missing:
        raise ValueError(f"missing required env vars for scope '{scope}': {', '.join(missing)}")
