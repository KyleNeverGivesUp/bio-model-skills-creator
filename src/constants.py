from datetime import datetime


def make_run_id(prefix: str = "pxr_activity") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


RUN_ID = make_run_id()
