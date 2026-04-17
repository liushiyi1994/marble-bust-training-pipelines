def required_env_vars(scope: str) -> list[str]:
    training = ["HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    if scope == "training":
        return training
    if scope == "runpod":
        return training + ["RUNPOD_API_KEY"]
    raise ValueError(f"unknown scope {scope}")
