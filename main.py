# main.py
import os, io, json, sys, time, logging, contextvars
from typing import List, Optional, Dict, Any
from uuid import uuid4

import dotenv
import requests
from PIL import Image
import imagehash

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from pydantic import BaseModel, Field

import httpx
from openai import OpenAI

# ──────────────────────── ENV / OpenAI client ────────────────────────────────
dotenv.load_dotenv()

# Evitar proxies heredados del entorno (causan errores con httpx)
for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(k, None)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-5-mini")  # o gpt-4o-mini-2024-07-18
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY no seteado")

# Cliente OpenAI con httpx controlado
client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(timeout=30.0))

# ───────────────────────── Logging estructurado ──────────────────────────────
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_ctx.get()
        return True

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("app")
logger.setLevel(LOG_LEVEL)
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(LOG_LEVEL)
_handler.addFilter(RequestIdFilter())
_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | req=%(request_id)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logger.handlers = [_handler]
logger.propagate = False

ENABLE_MODEL_DEBUG = os.getenv("ENABLE_MODEL_DEBUG", "false").lower() == "true"
MAX_DEBUG_CHARS = int(os.getenv("MAX_DEBUG_CHARS", "3000"))

def _trunc(s: str, n: int = MAX_DEBUG_CHARS) -> str:
    if s is None:
        return ""
    return s if len(s) <= n else s[:n] + f"...(truncated {len(s)-n} chars)"

# ─────────────────────────── FastAPI app ─────────────────────────────────────
app = FastAPI(
    title="Image Classifier & Sorter (Real Estate)",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("x-request-id") or str(uuid4())
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        try:
            logger.info(f"REQUEST {request.method} {request.url.path} q={request.url.query or '-'}")
            response = await call_next(request)
            dur_ms = (time.perf_counter() - start) * 1000
            logger.info(f"RESPONSE {request.method} {request.url.path} {response.status_code} in {dur_ms:.1f}ms")
            response.headers["X-Request-ID"] = rid
            return response
        except Exception as e:
            dur_ms = (time.perf_counter() - start) * 1000
            logger.exception(f"ERROR {request.method} {request.url.path} after {dur_ms:.1f}ms: {e}")
            raise
        finally:
            request_id_ctx.reset(token)

app.add_middleware(LoggingMiddleware)

# ─────────────────────────── Tipos (Pydantic) ────────────────────────────────
class ClassifyRequest(BaseModel):
    property_id: Optional[str] = None
    property_type: str = Field(..., description="apartment | house | land | office | commercial")
    image_urls: List[str] = Field(..., min_items=1)
    canonical_order: Optional[List[str]] = None

class ImageOut(BaseModel):
    id: str
    url: str
    primary_category: str
    secondary_labels: List[str] = []
    confidence: float = 0.0
    quality: Dict[str, Any] = {}
    flags: Dict[str, Any] = {}
    orientation: Optional[str] = None
    alt_text: Optional[str] = None
    tags: List[str] = []
    order_index: int
    is_cover: bool = False
    phash: Optional[str] = None
    size: Optional[Dict[str, int]] = None

class ClassifyResponse(BaseModel):
    version: str = "1.0"
    property_type: str
    canonical_order: List[str]
    images: List[ImageOut]
    cover_image_id: Optional[str] = None
    notes: Optional[str] = None

# ─────────────────────────── Utilidades ──────────────────────────────────────
CANONICALS = {
    "house": ["facade_exterior","living_room","kitchen","bedroom","bathroom","patio_garden","amenity_pool_gym","garage","laundry","hallway","view_window","floorplan","map","other"],
    "apartment": ["living_room","kitchen","bedroom","bathroom","balcony_terrace","amenity_pool_gym","building_common","facade_exterior","floorplan","map","other"],
    "land": ["facade_exterior","patio_garden","map","other"],
    "office": ["building_common","home_office","bathroom","kitchen","view_window","floorplan","map","other"],
    "commercial": ["facade_exterior","building_common","home_office","bathroom","floorplan","map","other"]
}

CATEGORIES = [
    "facade_exterior","building_common","living_room","dining_room","kitchen",
    "bedroom","bathroom","home_office","laundry","garage","balcony_terrace",
    "patio_garden","amenity_pool_gym","hallway","view_window","floorplan","map","other"
]

def get_canonical(pt: str, override: Optional[List[str]] = None) -> List[str]:
    if override and len(override) > 0:
        return override
    return CANONICALS.get(pt.lower().strip(), CANONICALS["apartment"])

def fetch_image_meta(url: str):
    """Descarga imagen y calcula pHash/orientación/tamaño (best effort)."""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        w, h = im.size
        orientation = "landscape" if w > h else "portrait" if h > w else "square"
        ph = imagehash.phash(im)
        return {"phash": str(ph), "size": {"w": w, "h": h}, "orientation": orientation}
    except Exception as e:
        logger.warning(f"fetch_image_meta failed for url={url[:120]}... err={e}")
        return None

def model_schema():
    return {
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "property_type": {"type": "string"},
            "canonical_order": {"type": "array", "items": {"type": "string"}},
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "url": {"type": "string"},
                        "primary_category": {"type": "string", "enum": CATEGORIES},
                        "secondary_labels": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number"},
                        "quality": {"type": "object"},
                        "flags": {"type": "object"},
                        "orientation": {"type": "string"},
                        "alt_text": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "order_index": {"type": "integer"},
                        "is_cover": {"type": "boolean"}
                    },
                    "required": ["id","url","primary_category","confidence","order_index","is_cover"]
                }
            },
            "cover_image_id": {"type": "string"},
            "notes": {"type": "string"}
        },
        "required": ["version","property_type","canonical_order","images"]
    }

SYSTEM_PROMPT = """Eres un asistente que clasifica y ordena fotos inmobiliarias.
Devuelves SOLO JSON válido según el esquema. Tareas:
1) Categoriza cada imagen en UNA de las categorías permitidas.
2) Detecta etiquetas útiles (tags), calidad básica y banderas (rostros, texto/teléfono, nsfw, watermark, floorplan).
3) Sugiere ALT text por accesibilidad.
4) Elige portada óptima y asigna order_index siguiendo el orden canónico entregado.
5) Si la confianza es baja, usa 'other' y deja al final.
No inventes información y no agregues campos fuera del esquema.
"""

def build_user_content(property_type: str, canonical: List[str], urls: List[str]):
    """Arma el contenido multimodal para Chat Completions (texto + imágenes)."""
    parts: List[Dict[str, Any]] = []
    intro = {
        "type": "text",
        "text": json.dumps({
            "property_type": property_type,
            "canonical_order": canonical,
            "categories_allowed": CATEGORIES,
            "rules": [
                "Dentro de cada categoría, prioriza mayor quality.score y sin personas/texto; duplicados al final",
                "Apartment: living_room primero; House: facade_exterior primero"
            ]
        }, ensure_ascii=False)
    }
    parts.append(intro)
    for u in urls:
        parts.append({"type": "image_url", "image_url": {"url": u}})
    return parts

# ─────────────────────────── Endpoints ───────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL,
        "openai_sdk": getattr(client, "__class__", type(client)).__name__,
        "httpx": httpx.__version__
    }

@app.post("/classify-images", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY missing")
        raise HTTPException(500, "Falta OPENAI_API_KEY")

    urls = [u.strip() for u in req.image_urls if isinstance(u, str) and u.strip()]
    if not urls:
        logger.warning("image_urls empty in request")
        raise HTTPException(400, "image_urls vacío")

    property_type = req.property_type.lower().strip()
    canonical = get_canonical(property_type, req.canonical_order)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(property_type, canonical, urls)}
    ]

    if ENABLE_MODEL_DEBUG:
        try:
            logger.debug("OPENAI messages=" + _trunc(json.dumps(messages, ensure_ascii=False)))
        except Exception:
            logger.debug("OPENAI messages=(unserializable)")

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=models := messages,  # mantiene una ref por si logeamos
            temperature=0,
            max_tokens=2000,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ImageOrdering", "schema": model_schema(), "strict": True}
            }
        )
        content = resp.choices[0].message.content
        if ENABLE_MODEL_DEBUG:
            logger.debug("OPENAI raw response=" + _trunc(content))
        data = json.loads(content)
    except Exception as e:
        logger.exception(f"OpenAI call failed: {e}")
        raise HTTPException(502, f"Error llamando a OpenAI: {e}")

    # Enriquecer con pHash/orientación/tamaño
    url_to_meta = {}
    for u in urls:
        meta = fetch_image_meta(u)
        if meta:
            url_to_meta[u] = meta

    for item in data.get("images", []):
        meta = url_to_meta.get(item.get("url"))
        if meta:
            item["phash"] = meta.get("phash")
            item["orientation"] = item.get("orientation") or meta.get("orientation")
            item["size"] = meta.get("size")

    try:
        out = ClassifyResponse(
            version=str(data.get("version", "1.0")),
            property_type=property_type,
            canonical_order=data.get("canonical_order", canonical),
            images=[ImageOut(**img) for img in data.get("images", [])],
            cover_image_id=data.get("cover_image_id"),
            notes=data.get("notes")
        )
    except Exception as e:
        logger.exception(f"Validation/serialization error: {e}; data={_trunc(json.dumps(data, ensure_ascii=False))}")
        raise HTTPException(500, "Error formateando la respuesta")

    logger.info(f"DONE classify images={len(out.images)} cover={out.cover_image_id or '-'}")
    return out

