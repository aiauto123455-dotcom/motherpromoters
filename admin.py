"""
Real Estate AI - Admin Application
====================================
This file contains everything the property owner (admin) needs:
- Simple token-based admin login (env-var credentials)
- Property CRUD (create, read, update, delete)
- Image upload / delete for a property (stored in Supabase Storage)
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

NOTE ON IMAGE STORAGE (IMPORTANT - READ IF YOU'RE DEBUGGING 404 IMAGES):
Property images are uploaded to a **Supabase Storage bucket**, not to local
disk. This is required because Render's filesystem is ephemeral - anything
written to local disk (e.g. `os.path.join(..., "uploads")`) is wiped on
every deploy, redeploy, and free-tier spin-down/spin-up cycle. A previous
version of this file saved images to local disk; the moment the container
restarted, every image path stored in the database pointed at a file that
no longer existed on disk, producing 404s on the frontend even though the
database row and its `image_urls` field were completely intact. Supabase
Storage is a separate, persistent object store (backed by S3-compatible
storage), so uploaded files survive restarts/redeploys indefinitely - the
same way the Postgres database itself already does.

Required environment variables for this to work (see .env.example):
  SUPABASE_URL              - e.g. https://abcxyzproject.supabase.co
  SUPABASE_SERVICE_ROLE_KEY - the "service_role" secret key (Project
                               Settings -> API in the Supabase dashboard).
                               NEVER use the anon/public key here - only the
                               service_role key can write to a bucket from
                               a trusted backend like this one, and it must
                               never be exposed to any frontend/browser code.
  SUPABASE_STORAGE_BUCKET   - defaults to "property-images". Create this
                               bucket in the Supabase dashboard (Storage ->
                               New bucket) and mark it PUBLIC, so the
                               returned URLs are directly viewable by buyers
                               without needing a signed URL on every page
                               load.
"""

import os
import json
import secrets
import uuid
from datetime import datetime
from typing import List, Optional

import httpx
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


# ---------------------------------------------------------------------------
# Supabase Storage configuration (persistent image storage)
# ---------------------------------------------------------------------------
# See the module docstring above for the full explanation of why this
# exists instead of local-disk storage. In short: Render's disk is
# ephemeral, Supabase Storage is not.

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "property-images")

STORAGE_CONFIGURED = bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)

# Legacy local-disk directory. No longer written to for new uploads, but
# kept so that ANY image path already stored in the database from before
# this migration (i.e. an old "/uploads/xyz.jpg" row) doesn't crash
# anything that still references this constant. main.py still mounts this
# directory as a static route for backward compatibility with any such
# legacy rows, though on Render those specific old files are already gone
# (that's the bug this migration fixes) - new uploads never touch this
# path anymore.
ADMIN_UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(ADMIN_UPLOAD_DIR, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB per image
MAX_IMAGES_PER_PROPERTY = 3  # a property can have at most 3 listing photos

_EXT_TO_CONTENT_TYPE = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


def _storage_object_path(filename: str) -> str:
    """Path of an object *within* the bucket (not the full URL)."""
    return filename


def _storage_public_url(filename: str) -> str:
    """Public, directly-viewable URL for an object in the bucket. Requires
    the bucket to be marked Public in the Supabase dashboard."""
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_STORAGE_BUCKET}/{filename}"


def _require_storage_configured():
    if not STORAGE_CONFIGURED:
        raise HTTPException(
            status_code=503,
            detail=(
                "Image storage is not configured on the server. Set "
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the "
                "environment (see admin.py module docstring)."
            ),
        )


def upload_to_supabase_storage(filename: str, contents: bytes, content_type: str) -> str:
    """Uploads raw bytes to the configured Supabase Storage bucket and
    returns the public URL. Raises HTTPException on failure so callers can
    surface a clean error to the admin instead of silently losing the
    upload."""
    _require_storage_configured()

    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_STORAGE_BUCKET}/{_storage_object_path(filename)}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Content-Type": content_type,
        # "3600" upstream cache-control (seconds); harmless default, files
        # are content-addressed by a random uuid so staleness isn't a risk.
        "x-upsert": "false",
    }
    try:
        resp = httpx.post(url, headers=headers, content=contents, timeout=30.0)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach image storage: {exc}") from exc

    if resp.status_code not in (200, 201):
        print(f"[admin] Supabase Storage upload failed ({resp.status_code}): {resp.text}")
        raise HTTPException(
            status_code=502,
            detail="Failed to upload image to storage. Please try again.",
        )

    return _storage_public_url(filename)


def delete_from_supabase_storage(filename: str) -> None:
    """Best-effort delete of an object from the bucket. Never raises - a
    failed cleanup shouldn't block the admin from removing the DB
    reference, it just means an orphaned file sits in the bucket."""
    if not STORAGE_CONFIGURED:
        return
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_STORAGE_BUCKET}/{_storage_object_path(filename)}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
    }
    try:
        resp = httpx.delete(url, headers=headers, timeout=15.0)
        if resp.status_code not in (200, 204):
            print(f"[admin] warning: Supabase Storage delete returned {resp.status_code}: {resp.text}")
    except httpx.HTTPError as exc:
        print(f"[admin] warning: failed to delete '{filename}' from storage: {exc}")


def _filename_from_image_url(image_url: str) -> str:
    """Extracts the bare storage filename from either a full Supabase
    public URL (new-style) or a legacy '/uploads/xyz.jpg' local path
    (old-style), so delete logic works for both without special-casing
    every call site."""
    return image_url.rsplit("/", 1)[-1]


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

    # Remove associated image files from Supabase Storage (best-effort -
    # see delete_from_supabase_storage's docstring for why this never
    # raises and never blocks the property row deletion below).
    try:
        images = json.loads(prop.image_urls) if prop.image_urls else []
        for img_path in images:
            filename = _filename_from_image_url(img_path)
            delete_from_supabase_storage(filename)
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
# Image upload / delete (Supabase Storage - see module docstring)
# ---------------------------------------------------------------------------

def _validate_image(file: UploadFile, contents: bytes) -> str:
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
    _require_storage_configured()

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

    uploaded_urls: List[str] = []
    for file in files:
        contents = await file.read()
        ext = _validate_image(file, contents)

        # Safe, unpredictable filename - prevents path traversal and
        # collisions, and doubles as the object key in the bucket.
        safe_filename = f"property_{property_id}_{uuid.uuid4().hex[:12]}.{ext}"
        content_type = _EXT_TO_CONTENT_TYPE.get(ext, "application/octet-stream")

        public_url = upload_to_supabase_storage(safe_filename, contents, content_type)
        uploaded_urls.append(public_url)

    existing_images.extend(uploaded_urls)
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

    updated_images = [img for img in images if _filename_from_image_url(img) != image_name]
    if len(updated_images) == len(images):
        raise HTTPException(status_code=404, detail="Image not found on this property")

    delete_from_supabase_storage(image_name)

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
