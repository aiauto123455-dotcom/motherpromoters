"""
Real Estate AI - Admin Application
====================================
This file contains everything the property owner (admin) needs:
- Simple token-based admin login (env-var credentials)
- Property CRUD (create, read, update, delete)
- Image upload / delete for a property
- Dashboard stats
- Leads listing

NOTE ON AUTH (MVP-LEVEL):
This uses a lightweight bearer-token scheme suitable for a small MVP with a
single admin user: on successful login we issue a random opaque token, keep
it in memory on the server, and require it on every admin request via the
`Authorization: Bearer <token>` header. This is NOT full JWT/session auth
(no expiry rotation, no refresh tokens, no persistence across restarts).
For production, replace this with proper JWT or server-side sessions plus
HTTPS-only secure cookies.
"""

import os
import json
import secrets
import shutil
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Header
from pydantic import BaseModel, Field

from sqlalchemy.orm import Session

# Import shared DB pieces from main.py's module-level definitions.
# main.py imports admin_router from this file AFTER these are defined,
# so this import is safe (no circular execution issue at import time
# because main.py does the admin import last, and this module only
# needs main's DB objects, which exist by then... to avoid any fragile
# ordering, we instead define our own lightweight DB access below using
# the same engine/session factory pattern kept in sync with main.py).

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import NullPool

# ---------------------------------------------------------------------------
# Reuse the exact same database configuration as main.py
# ---------------------------------------------------------------------------
#
# IMPORTANT: this mirrors main.py's DB setup exactly (same URL normalization,
# same pgbouncer/Supavisor handling, same conditional connect_args) because
# admin.py builds its OWN engine rather than importing main's. If these two
# ever drift apart, admin.py can silently misconfigure its connection the
# moment DATABASE_URL points somewhere main.py handles differently - so any
# change to main.py's DB setup should be mirrored here too.
#
# FIX: the previous version read `os.getenv("DATABASE_URL")` with NO
# default, unlike main.py's `os.getenv("DATABASE_URL", "sqlite:///...")`.
# If DATABASE_URL was ever unset (e.g. local dev without a .env, or a
# misconfigured deploy), RAW_DATABASE_URL was `None`, and
# `None.startswith("postgres://")` raised AttributeError at import time -
# which crashed the entire app on startup, since main.py imports
# admin_router from this module. The default below matches main.py exactly
# so admin.py can never crash on import for this reason, and both modules
# fall back to the same local SQLite file in the same circumstances.
RAW_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./realestate.db")

if RAW_DATABASE_URL.startswith("postgres://"):
    # SQLAlchemy 1.4+/2.x only accepts "postgresql://" - Supabase/Render/
    # Heroku-style providers commonly still hand out the old
    # "postgres://" scheme.
    DATABASE_URL = RAW_DATABASE_URL.replace("postgres://", "postgresql://", 1)
else:
    DATABASE_URL = RAW_DATABASE_URL

IS_SQLITE = DATABASE_URL.startswith("sqlite")

# Same pooled-Supabase (Supavisor/pgbouncer, transaction mode) detection as
# main.py - see the long comment there for why this needs NullPool and
# disabled statement caching. Kept as a literal copy rather than an import
# to avoid any import-order coupling between the two modules.
_looks_pooled = (":6543" in DATABASE_URL) or ("pooler.supabase.com" in DATABASE_URL)
IS_POOLED_PGBOUNCER = os.getenv(
    "SUPABASE_POOLED", "true" if _looks_pooled else "false"
).strip().lower() in ("1", "true", "yes")

_engine_kwargs = {}
if IS_SQLITE:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
elif IS_POOLED_PGBOUNCER:
    _engine_kwargs["poolclass"] = NullPool
    _engine_kwargs["connect_args"] = {
        "prepare_threshold": None,
        "options": "-c statement_timeout=30000",
    }
    _engine_kwargs["pool_pre_ping"] = True
else:
    # Direct Supabase connection (port 5432) or any other plain Postgres.
    # Avoids "SSL connection has been closed unexpectedly" errors from
    # managed Postgres providers dropping idle connections.
    _engine_kwargs["pool_pre_ping"] = True

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utcnow() -> datetime:
    return datetime.utcnow()


class Property(Base):
    __tablename__ = "properties"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    property_type = Column(String, nullable=False)
    location = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    area_sqft = Column(Float, nullable=True)
    plot_size = Column(String, nullable=True)
    bedrooms = Column(Integer, nullable=True)
    bathrooms = Column(Integer, nullable=True)
    facing = Column(String, nullable=True)
    approval = Column(String, nullable=True)
    road_width = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="available")
    image_urls = Column(Text, nullable=False, default="[]")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, nullable=True)
    name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    interested_property_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=utcnow)


# Tables already created by main.py's Base.metadata.create_all, but running
# it again here is a harmless no-op safeguard (e.g. if admin.py were ever
# imported/run standalone).
Base.metadata.create_all(bind=engine)


ADMIN_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(ADMIN_UPLOAD_DIR, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB per image
MAX_IMAGES_PER_PROPERTY = 3  # a property can have at most 3 listing photos

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change_this_password")

# In-memory token store: {token: username}. Simple and fine for a single-
# admin MVP on one server process. Lost on restart (admin just logs in again).
_active_tokens: dict = {}

admin_router = APIRouter(prefix="/api/admin", tags=["admin"])


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    username: str
    password: str


def require_admin(authorization: Optional[str] = Header(None)) -> str:
    """Dependency that protects every admin route. Expects:
    Authorization: Bearer <token>
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    username = _active_tokens.get(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired session. Please log in again.")
    return username


@admin_router.post("/login")
def admin_login(payload: LoginRequest):
    if payload.username == ADMIN_USERNAME and payload.password == ADMIN_PASSWORD:
        token = secrets.token_hex(32)
        _active_tokens[token] = payload.username
        return {"status": "ok", "token": token}
    raise HTTPException(status_code=401, detail="Invalid username or password")


@admin_router.post("/logout")
def admin_logout(username: str = Depends(require_admin), authorization: Optional[str] = Header(None)):
    token = authorization.removeprefix("Bearer ").strip()
    _active_tokens.pop(token, None)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

VALID_PROPERTY_TYPES = {"Plot", "House", "Apartment", "Villa", "Commercial"}
VALID_STATUSES = {"available", "sold", "reserved"}


class PropertyIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    property_type: str
    location: str = Field(..., min_length=1, max_length=200)
    price: float = Field(..., gt=0)
    area_sqft: Optional[float] = Field(None, ge=0)
    plot_size: Optional[str] = None
    bedrooms: Optional[int] = Field(None, ge=0, le=50)
    bathrooms: Optional[int] = Field(None, ge=0, le=50)
    facing: Optional[str] = None
    approval: Optional[str] = None
    road_width: Optional[str] = None
    description: Optional[str] = None
    status: str = "available"

    def validated(self) -> "PropertyIn":
        if self.property_type not in VALID_PROPERTY_TYPES:
            raise HTTPException(status_code=422, detail=f"property_type must be one of {sorted(VALID_PROPERTY_TYPES)}")
        if self.status not in VALID_STATUSES:
            raise HTTPException(status_code=422, detail=f"status must be one of {sorted(VALID_STATUSES)}")
        return self


class PropertyOut(BaseModel):
    id: int
    name: str
    property_type: str
    location: str
    price: float
    area_sqft: Optional[float] = None
    plot_size: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    facing: Optional[str] = None
    approval: Optional[str] = None
    road_width: Optional[str] = None
    description: Optional[str] = None
    status: str
    images: List[str] = []
    created_at: str
    updated_at: str


class StatusUpdate(BaseModel):
    status: str


def _sanitize_text(value: Optional[str]) -> Optional[str]:
    """Basic sanitization: strip and cap length to avoid stray HTML/script
    content being stored verbatim. The frontend also escapes on render."""
    if value is None:
        return None
    return value.strip()[:5000]


def property_to_out(p: Property) -> PropertyOut:
    try:
        images = json.loads(p.image_urls) if p.image_urls else []
    except json.JSONDecodeError:
        images = []
    return PropertyOut(
        id=p.id,
        name=p.name,
        property_type=p.property_type,
        location=p.location,
        price=p.price,
        area_sqft=p.area_sqft,
        plot_size=p.plot_size,
        bedrooms=p.bedrooms,
        bathrooms=p.bathrooms,
        facing=p.facing,
        approval=p.approval,
        road_width=p.road_width,
        description=p.description,
        status=p.status,
        images=images,
        created_at=p.created_at.isoformat() if p.created_at else "",
        updated_at=p.updated_at.isoformat() if p.updated_at else "",
    )


# ---------------------------------------------------------------------------
# Property CRUD
# ---------------------------------------------------------------------------

@admin_router.get("/properties", response_model=List[PropertyOut])
def list_properties(username: str = Depends(require_admin), db: Session = Depends(get_db)):
    props = db.query(Property).order_by(Property.created_at.desc()).all()
    return [property_to_out(p) for p in props]


@admin_router.post("/properties", response_model=PropertyOut)
def create_property(
    payload: PropertyIn,
    username: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    payload = payload.validated()
    prop = Property(
        name=_sanitize_text(payload.name),
        property_type=payload.property_type,
        location=_sanitize_text(payload.location),
        price=payload.price,
        area_sqft=payload.area_sqft,
        plot_size=_sanitize_text(payload.plot_size),
        bedrooms=payload.bedrooms,
        bathrooms=payload.bathrooms,
        facing=_sanitize_text(payload.facing),
        approval=_sanitize_text(payload.approval),
        road_width=_sanitize_text(payload.road_width),
        description=_sanitize_text(payload.description),
        status=payload.status,
        image_urls="[]",
        created_at=utcnow(),
        updated_at=utcnow(),
    )
    db.add(prop)
    db.commit()
    db.refresh(prop)
    return property_to_out(prop)


@admin_router.get("/properties/{property_id}", response_model=PropertyOut)
def get_property(property_id: int, username: str = Depends(require_admin), db: Session = Depends(get_db)):
    prop = db.query(Property).filter(Property.id == property_id).first()
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")
    return property_to_out(prop)


@admin_router.put("/properties/{property_id}", response_model=PropertyOut)
def update_property(
    property_id: int,
    payload: PropertyIn,
    username: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    payload = payload.validated()
    prop = db.query(Property).filter(Property.id == property_id).first()
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")

    prop.name = _sanitize_text(payload.name)
    prop.property_type = payload.property_type
    prop.location = _sanitize_text(payload.location)
    prop.price = payload.price
    prop.area_sqft = payload.area_sqft
    prop.plot_size = _sanitize_text(payload.plot_size)
    prop.bedrooms = payload.bedrooms
    prop.bathrooms = payload.bathrooms
    prop.facing = _sanitize_text(payload.facing)
    prop.approval = _sanitize_text(payload.approval)
    prop.road_width = _sanitize_text(payload.road_width)
    prop.description = _sanitize_text(payload.description)
    prop.status = payload.status
    prop.updated_at = utcnow()

    db.add(prop)
    db.commit()
    db.refresh(prop)
    return property_to_out(prop)


@admin_router.delete("/properties/{property_id}")
def delete_property(property_id: int, username: str = Depends(require_admin), db: Session = Depends(get_db)):
    prop = db.query(Property).filter(Property.id == property_id).first()
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")

    # Remove associated image files from disk
    try:
        images = json.loads(prop.image_urls) if prop.image_urls else []
        for img_path in images:
            filename = os.path.basename(img_path)
            full_path = os.path.join(ADMIN_UPLOAD_DIR, filename)
            if os.path.exists(full_path):
                os.remove(full_path)
    except Exception as exc:
        print(f"[admin] warning: failed to remove image files for property {property_id}: {exc}")

    db.delete(prop)
    db.commit()
    return {"status": "ok", "deleted_id": property_id}


@admin_router.patch("/properties/{property_id}/status", response_model=PropertyOut)
def update_property_status(
    property_id: int,
    payload: StatusUpdate,
    username: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if payload.status not in VALID_STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(VALID_STATUSES)}")

    prop = db.query(Property).filter(Property.id == property_id).first()
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")

    prop.status = payload.status
    prop.updated_at = utcnow()
    db.add(prop)
    db.commit()
    db.refresh(prop)
    return property_to_out(prop)


# ---------------------------------------------------------------------------
# Image upload / delete
# ---------------------------------------------------------------------------

def _validate_image(file: UploadFile, contents: bytes):
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{ext}'. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
        )
    if len(contents) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="Image too large. Max size is 5MB.")
    return ext


@admin_router.post("/properties/{property_id}/images", response_model=PropertyOut)
async def upload_property_images(
    property_id: int,
    files: List[UploadFile] = File(...),
    username: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    prop = db.query(Property).filter(Property.id == property_id).first()
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")

    try:
        existing_images = json.loads(prop.image_urls) if prop.image_urls else []
    except json.JSONDecodeError:
        existing_images = []

    if not files:
        raise HTTPException(status_code=400, detail="No image files were provided.")

    if len(existing_images) + len(files) > MAX_IMAGES_PER_PROPERTY:
        remaining = max(MAX_IMAGES_PER_PROPERTY - len(existing_images), 0)
        raise HTTPException(
            status_code=400,
            detail=(
                f"A property can have at most {MAX_IMAGES_PER_PROPERTY} images. "
                f"This property already has {len(existing_images)}, so you can add "
                f"{remaining} more."
            ),
        )

    for file in files:
        contents = await file.read()
        ext = _validate_image(file, contents)

        # Safe, unpredictable filename - prevents path traversal and collisions.
        safe_filename = f"property_{property_id}_{uuid.uuid4().hex[:12]}.{ext}"
        dest_path = os.path.join(ADMIN_UPLOAD_DIR, safe_filename)

        with open(dest_path, "wb") as f:
            f.write(contents)

        existing_images.append(f"/uploads/{safe_filename}")

    prop.image_urls = json.dumps(existing_images)
    prop.updated_at = utcnow()
    db.add(prop)
    db.commit()
    db.refresh(prop)
    return property_to_out(prop)


@admin_router.delete("/properties/{property_id}/images/{image_name}")
def delete_property_image(
    property_id: int,
    image_name: str,
    username: str = Depends(require_admin),
    db: Session = Depends(get_db),
):
    # Guard against path traversal: only allow a bare filename, no slashes.
    if "/" in image_name or ".." in image_name or "\\" in image_name:
        raise HTTPException(status_code=400, detail="Invalid image name")

    prop = db.query(Property).filter(Property.id == property_id).first()
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found")

    try:
        images = json.loads(prop.image_urls) if prop.image_urls else []
    except json.JSONDecodeError:
        images = []

    updated_images = [img for img in images if os.path.basename(img) != image_name]
    if len(updated_images) == len(images):
        raise HTTPException(status_code=404, detail="Image not found on this property")

    full_path = os.path.join(ADMIN_UPLOAD_DIR, image_name)
    if os.path.exists(full_path):
        os.remove(full_path)

    prop.image_urls = json.dumps(updated_images)
    prop.updated_at = utcnow()
    db.add(prop)
    db.commit()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Stats + Leads
# ---------------------------------------------------------------------------

@admin_router.get("/stats")
def get_stats(username: str = Depends(require_admin), db: Session = Depends(get_db)):
    total = db.query(Property).count()
    available = db.query(Property).filter(Property.status == "available").count()
    sold = db.query(Property).filter(Property.status == "sold").count()
    reserved = db.query(Property).filter(Property.status == "reserved").count()
    total_leads = db.query(Lead).count()
    return {
        "total_properties": total,
        "available": available,
        "sold": sold,
        "reserved": reserved,
        "total_leads": total_leads,
    }


@admin_router.get("/leads")
def get_leads(username: str = Depends(require_admin), db: Session = Depends(get_db)):
    leads = db.query(Lead).order_by(Lead.created_at.desc()).all()
    result = []
    for lead in leads:
        property_name = None
        if lead.interested_property_id:
            prop = db.query(Property).filter(Property.id == lead.interested_property_id).first()
            property_name = prop.name if prop else None
        result.append(
            {
                "id": lead.id,
                "session_id": lead.session_id,
                "name": lead.name,
                "phone": lead.phone,
                "interested_property_id": lead.interested_property_id,
                "interested_property_name": property_name,
                "created_at": lead.created_at.isoformat() if lead.created_at else "",
            }
        )
    return result
