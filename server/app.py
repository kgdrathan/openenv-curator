"""
FastAPI application for the Curator Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from ..models import CuratorAction, CuratorObservation
    from .curator_environment import CuratorEnvironment
except (ImportError, ModuleNotFoundError):
    from models import CuratorAction, CuratorObservation
    from server.curator_environment import CuratorEnvironment


app = create_app(
    CuratorEnvironment,
    CuratorAction,
    CuratorObservation,
    env_name="curator",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
