from pgvector.sqlalchemy import Vector
from sqlalchemy import Integer, String, Text, create_engine, text 
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://careerpilot:careerpilot@localhost:5433/careerpilot",
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(engine)

class Base(DeclarativeBase):
    pass

class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    company: Mapped[str] = mapped_column(String(200), default="")
    description: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(384))

def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)