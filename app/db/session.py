"""
Async database engine and session factory.
Using asyncpg driver for non-blocking PostgreSQL I/O.
"""

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.config import get_settings
from app.db.models import Base

settings = get_settings()

# Engine: single instance shared across the app
# pool_size=10 handles 10 concurrent DB connections
# max_overflow=20 allows burst up to 30 connections
engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    echo=False,  # Set True locally to see SQL — never in prod
)

# Session factory — call this to get an async session
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit (safer with async)
)


async def init_db():
    """
    Create all tables on startup.
    In production, replace this with Alembic migrations.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """
    FastAPI dependency injection target.
    Usage: db: AsyncSession = Depends(get_db)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise