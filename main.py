# main.py
import os, io, json, base64, time, sys, logging, contextvars
from typing import List, Dict, Any, Optional
from uuid import uuid4

import dotenv
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import httpx
from openai import OpenAI

# ────────── ENV & OpenAI client ──────────
dotenv.load_dotenv()
for k in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
    os.environ.pop(k, None)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-5-mini")  # gpt-4o-mini-2024-07-18 si querés ultra barato

client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(timeout=30.0))

# ────────── Logging con request-id ──────────
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_ctx.get()
        return True

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("app"); logger.setLevel(LOG_LEVEL)
_handler = logging.StreamHandler(sys.stdout); _handler.setLevel(LOG_LEVEL)
_handler.addFilter(RequestIdFilter())
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | req=%(request_id)s | %(message)s",
                                        datefmt="%Y-%m-%d %H:%M:%S"))
logger.handlers = [_handler]; logger.propagate = False

# ────────── App ──────────
app = FastAPI(
    title="Image Classifier & Sorter (Real Estate)",
    docs_url="/docs", redoc_url="/redoc", openapi_url="/openapi.json"
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ────────── Taxonomía & esquema ──────────
CATEGORIES = [
    "facade_exterior","building_common","living_room","dining_room","kitchen",
    "bedroom","bathroom","home_office","laundry","garage","balcony_terrace",
    "patio_garden","amenity_pool_gym","hallway","view_window","floorplan","map","other"
]
CANONICALS = {
    "house": ["facade_exterior","living_room","kitchen","bedroom","bathroom","patio_garden","amenity_pool_gym","garage","laundry","hallway","view_window","floorplan","map","other"],
    "apartment": ["living_room","kitchen","bedroom","bathroom","balcony_terrace","amenity_pool_gym","building_common","facade_exterior","floorplan","map","other"],
    "land": ["facade_exterior","patio_garden","map","other"],
    "office": ["building_common","home_office","bathroom","kitchen","view_window","floorplan","map","other"],
    "commercial": ["facade_exterior","building_common","home_office","bathroom","floorplan","map","other"]
}
def get_canonical(pt: str, override: Optional[List[str]]=None) -> List[str]:
    return override if override else CANONICALS.get(pt.lower().strip(), CANONICALS["apartment"])

def model_schema():
    return {
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "property_type": {"type": "string"},
            "canonical_order": {"type": "array","items":{"type":"string"}},
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type":"string"},
                        "url": {"type":"string"},
                        "primary_category": {"type":"string","enum": CATEGORIES},
                        "secondary_labels": {"type":"array","items":{"type":"string"}},
                        "confidence": {"type":"number"},
                        "quality": {"type":"object"},
                        "flags": {"type":"object"},
                        "orientation": {"type":"string"},
                        "alt_text": {"type":"string"},
                        "tags": {"type":"array","items":{"type":"string"}},
                        "order_index": {"type":"integer"},
                        "is_cover": {"type":"boolean"}
                    },
                    "required": ["id","url","primary_category","confidence","order_index","is_cover"]
                }
            },
            "cover_image_id": {"type":"string"},
            "notes": {"type":"string"}
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

def build_user_content(property_type: str, canonical: List[str], image_parts: List[Dict[str, Any]]):
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
    return [intro] + image_parts

# ────────── Helpers ──────────
def file_to_image_part(f: UploadFile) -> Optional[Dict[str, Any]]:
    """Devuelve un bloque para Chat Completions (image_url con data URI) o None."""
    if not f:
        return None
    ct = (f.content_type or "").lower()
    try:
        data = f.file.read()
    except Exception:
        return None
    if not data or not ct.startswith("image/"):
        return None
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{ct};base64,{b64}"}}

def collect_images_from_form(form) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    # img1..img50
    for i in range(1, 51):
        f = form.get(f"img{i}")
        if isinstance(f, UploadFile):
            p = file_to_image_part(f)
            if p: parts.append(p)
    # también aceptar cualquier key que empiece por 'img' por si agregás más
    if not parts:
        for key, val in form.items():
            if isinstance(val, UploadFile) and str(key).lower().startswith("img"):
                p = file_to_image_part(val)
                if p: parts.append(p)
    return parts

# ────────── Endpoints ──────────
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "openai_sdk": "OpenAI", "httpx": httpx.__version__}

@app.post("/classify-upload")
async def classify_upload(request: Request):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "Falta OPENAI_API_KEY")

    form = await request.form()
    property_type = (form.get("property_type") or "apartment").strip().lower()

    # canonical_order opcional (puede venir como JSON string o CSV)
    canonical_override: Optional[List[str]] = None
    raw = form.get("canonical_order")
    if raw:
        try:
            if raw.strip().startswith("["):
                canonical_override = json.loads(raw)
            else:
                canonical_override = [s.strip() for s in raw.split(",") if s.strip()]
        except Exception:
            canonical_override = None

    image_parts = collect_images_from_form(form)
    if not image_parts:
        raise HTTPException(400, "Envía al menos una imagen (img1..img50) como archivo.")

    canonical = get_canonical(property_type, canonical_override)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(property_type, canonical, image_parts)}
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=2000,
            response_format={
                "type": "json_schema",
                "json_schema": {"name":"ImageOrdering","schema": model_schema(), "strict": True}
            }
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.exception(f"OpenAI error: {e}")
        raise HTTPException(502, f"Error llamando a OpenAI: {e}")

    # Forzamos version y property_type por si el modelo no los pone
    data.setdefault("version", "1.0")
    data["property_type"] = property_type
    data.setdefault("canonical_order", canonical)

    return JSONResponse(content=data)
