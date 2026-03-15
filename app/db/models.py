"""
SQLAlchemy ORM models — these define the PostgreSQL schema.

Design principles:
- UUIDs as primary keys (safe for distributed systems, no integer collision)
- All costs stored as NUMERIC for financial precision (never float in a DB for money)
- Indexes on columns we'll query frequently (timestamp, caller_id, routed_model)
- Feedback linked to requests via foreign key — enforces referential integrity
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean,
    DateTime, Text, Numeric, ForeignKey, Index
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    """
    One row per /complete call.
    This is the core audit trail and cost ledger.
    """
    __tablename__ = "request_logs"

    # Identity
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    caller_id = Column(String(100), nullable=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Input
    prompt = Column(Text, nullable=False)

    # Classification
    difficulty_tags = Column(ARRAY(String), nullable=False)
    classifier_confidence = Column(Float, nullable=False)

    # Routing
    routed_model = Column(String(50), nullable=False, index=True)
    routing_reason = Column(Text, nullable=False)

    # Output
    response_text = Column(Text, nullable=False)

    # Token accounting
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)

    # Cost accounting — Numeric(12,6) = up to $999,999.999999 precision
    actual_cost_usd = Column(Numeric(12, 6), nullable=False)
    baseline_gpt4o_cost_usd = Column(Numeric(12, 6), nullable=False)
    cost_saved_usd = Column(Numeric(12, 6), nullable=False)

    # Performance
    latency_ms = Column(Float, nullable=False)

    # Relationship to feedback (one request can have one feedback)
    feedback = relationship("FeedbackLog", back_populates="request", uselist=False)

    # Compound index: common query is "show me all GPT-4o calls this week"
    __table_args__ = (
        Index("ix_routed_model_timestamp", "routed_model", "timestamp"),
    )


class FeedbackLog(Base):
    """
    One row per /feedback call.
    Linked to RequestLog by foreign key.
    """
    __tablename__ = "feedback_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(
        UUID(as_uuid=True),
        ForeignKey("request_logs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,    # one feedback per request, enforced at DB level
        index=True
    )
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # Human signal
    rating = Column(Integer, nullable=False)
    underpowered = Column(Boolean, default=False, nullable=False)
    overkill = Column(Boolean, default=False, nullable=False)
    comment = Column(Text, nullable=True)

    # Back-reference
    request = relationship("RequestLog", back_populates="feedback")


class RoutingAdjustment(Base):
    """
    Tracks when adaptive routing changes a rule.
    Gives us an audit trail of how the system learned over time.
    Think of this as the 'git history' of routing decisions.
    """
    __tablename__ = "routing_adjustments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # What changed
    difficulty_tag = Column(String(100), nullable=False, index=True)
    previous_model = Column(String(50), nullable=False)
    new_model = Column(String(50), nullable=False)
    reason = Column(Text, nullable=False)  # e.g., "underpowered_rate exceeded 0.3 threshold"

    # What triggered it
    trigger_feedback_count = Column(Integer, nullable=False)
    underpowered_rate = Column(Float, nullable=True)
    overkill_rate = Column(Float, nullable=True)