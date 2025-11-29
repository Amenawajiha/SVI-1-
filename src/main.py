"""Main FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware

from .chat import chat_router
from .config.settings import settings
from .utils import create_log_file, logger

create_log_file("svi_chat_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting application...")
    yield
    logger.info("Shutting down application...")


# Create FastAPI application
app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.exception("Unhandled error: %s - %s", exc, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred."},
    )


class WebSocketCORSMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling CORS preflight requests for WebSocket connections.

    This middleware adds the 'Access-Control-Allow-Origin' header to the WebSocket
    request if the 'Origin' header is present in the request headers. This is necessary
    to allow cross-origin WebSocket connections.

    Methods:
        dispatch(request, call_next): Asynchronously handles the incoming request and
        modifies the headers for WebSocket connections to include CORS headers if needed.
    """

    async def dispatch(self, request, call_next):
        if "websocket" in request.scope["type"]:
            if "origin" in request.headers:
                request.scope["headers"].append(
                    (b"Access-Control-Allow-Origin", request.headers["origin"].encode())
                )
        return await call_next(request)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(WebSocketCORSMiddleware)

# Include routers
app.include_router(chat_router)


@app.get("/debug/routes")
async def debug_routes():
    """Debug endpoint to list all registered routes."""
    routes = []
    for route in app.routes:
        if isinstance(route, APIRoute):
            routes.append(
                {
                    "path": route.path,
                    "name": route.name,
                    "methods": list(route.methods),
                    "endpoint": (
                        route.endpoint.__name__
                        if hasattr(route.endpoint, "__name__")
                        else str(route.endpoint)
                    ),
                }
            )
        elif hasattr(route, "endpoint") and hasattr(route.endpoint, "__name__"):
            routes.append(
                {
                    "path": route.path,
                    "name": route.endpoint.__name__,
                    "methods": getattr(route, "methods", []),
                    "endpoint": str(route.endpoint),
                }
            )
    return {"routes": routes}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
