import os, sys, json, base64, logging, contextvars
from typing import List, Dict, Any, Optional

import dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import httpx
from openai import OpenAI, BadRequestError

# ──────────────────────────────── ENV / OpenAI ────────────────────────────────
dotenv.load_dotenv()
for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(k, None)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini-2024-07-18")  # ← usamos el que ya te funcionaba

client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=httpx.Client(timeout=30.0),
)

# ──────────────────────────────── Logging ─────────────────────────────────────
request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_ctx.get()
        return True

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logger = logging.getLogger("app")
logger.setLevel(LOG_LEVEL)
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(LOG_LEVEL)
_handler.addFilter(RequestIdFilter())
_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)s | req=%(request_id)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logger.handlers = [_handler]
logger.propagate = False

# ──────────────────────────────── FastAPI ─────────────────────────────────────
app = FastAPI(
    title="Image Classifier & Sorter (Real Estate)",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# ──────────────────────────────── Taxonomía ───────────────────────────────────
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
    "commercial": ["facade_exterior","building_common","home_office","bathroom","floorplan","map","other"],
}
def get_canonical(pt: str, override: Optional[List[str]] = None) -> List[str]:
    return override if override else CANONICALS.get(pt.lower().strip(), CANONICALS["apartment"])

def model_schema() -> Dict[str, Any]:
    # Esquema minimal: categoría, orden, portada y etiquetas secundarias
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "version": {"type": "string"},
            "property_type": {"type": "string"},
            # Orden canónico que el modelo usó para decidir el orden final por categoría
            "canonical_order": {"type": "array", "items": {"type": "string"}},
            "images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id":  {"type": "string"},
                        "url": {"type": "string"},
                        "primary_category": {
                            "type": "string",
                            "enum": CATEGORIES  # usa tu lista de categorías (sala, comedor, baño, fachada, etc.)
                        },
                        "secondary_labels": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "order_index": {"type": "integer"},   # índice final (0..n) agrupando por categoría
                        "is_cover":    {"type": "boolean"}    # true para la portada
                    },
                    # strict:true → todas las keys declaradas deben estar en required
                    "required": [
                        "id",
                        "url",
                        "primary_category",
                        "secondary_labels",
                        "order_index",
                        "is_cover"
                    ]
                }
            },
            "cover_image_id": {"type": "string"}
        },
        "required": ["version", "property_type", "canonical_order", "images"]
    }

SYSTEM_PROMPT = """
Eres un asistente que clasifica y ordena fotos inmobiliarias.

Tareas:
1) Asigna 'primary_category' a cada imagen usando este set cerrado (en español, minúsculas): 
   sala, comedor, cocina, dormitorio, baño, pasillo, balcón, terraza, lavadero, quincho, patio,
   jardín, fachada, garaje, amenities, plano, escritorio/oficina, depósito, otros.
2) Agrega 'secondary_labels' (palabras clave cortas) si aplican, ej: 'vista a la ciudad', 
   'isla de cocina', 'ducha', 'toilette', 'placard', 'parrilla', 'piscina', 'doble altura', etc.
3) Ordena las imágenes agrupándolas por categoría y siguiendo el orden canónico del tipo de propiedad:
   - departamento: sala/living → comedor → cocina → dormitorios → baños → balcón/terraza → lavadero → amenities → garaje → fachada → plano → otros
   - casa: fachada → sala/living → comedor → cocina → dormitorios → baños → patio/jardín/quincho → lavadero → garaje → amenities → plano → otros
   - oficina/local: fachada → sala principal → ambientes secundarios → baños → kitchenette → amenities → plano → otros
   Devuelve ese orden en 'canonical_order' y usa 'order_index' (0..n) para cada imagen.
4) Elige 'is_cover' = true en una sola imagen (la portada) priorizando: 
   a) categoría atractiva (sala/living o fachada), 
   b) composición centrada, 
   c) buena iluminación aparente, 
   d) ausencia de personas/textos/marcas visibles, 
   e) que resuma mejor el valor de la propiedad.
   Guarda su id en 'cover_image_id'.

Formato de salida: sólo JSON que cumpla el esquema; no inventes categorías fuera del enum.
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

# ──────────────────────────────── Helpers (files) ─────────────────────────────
def is_upload_like(v: Any) -> bool:
    return hasattr(v, "filename") and hasattr(v, "file")

def file_to_image_part(upload_obj) -> Optional[Dict[str, Any]]:
    if not is_upload_like(upload_obj):
        return None
    ct = (getattr(upload_obj, "content_type", None) or "").lower()
    try:
        raw = upload_obj.file.read()
    except Exception:
        return None
    if not raw or not ct.startswith("image/"):
        return None
    b64 = base64.b64encode(raw).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{ct};base64,{b64}"}}

def collect_images_from_form(form) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    # img1..img50 (como pediste)
    for i in range(1, 51):
        v = form.get(f"img{i}")
        if is_upload_like(v):
            p = file_to_image_part(v)
            if p: parts.append(p)
    # fallback por si vienen con otras keys
    if not parts:
        for k, v in form.items():
            if str(k).lower().startswith("img") and is_upload_like(v):
                p = file_to_image_part(v)
                if p: parts.append(p)
    return parts

# ──────────────────────────────── Endpoints ───────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL}

@app.post("/debug-upload")
async def debug_upload(request: Request):
    form = await request.form()
    items = []
    for k, v in form.items():
        if is_upload_like(v):
            items.append({"key": k, "type": "UploadFile", "filename": v.filename, "content_type": getattr(v, "content_type", None)})
        else:
            try:
                val = str(v)
            except Exception:
                val = f"<{type(v).__name__}>"
            items.append({"key": k, "type": type(v).__name__, "value": val[:120]})
    logger.debug("DEBUG UPLOAD FORM: " + json.dumps(items, ensure_ascii=False))
    return {"received": items}

@app.post("/classify-upload")
async def classify_upload(request: Request):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "Falta OPENAI_API_KEY")

    form = await request.form()

    # Log resumido de lo recibido
    log_items = []
    for k, v in form.items():
        if is_upload_like(v):
            log_items.append({"k": k, "type": "UploadFile", "filename": v.filename, "content_type": getattr(v, "content_type", None)})
        else:
            try:
                val = str(v)
            except Exception:
                val = f"<{type(v).__name__}>"
            log_items.append({"k": k, "type": type(v).__name__, "value": val[:120]})
    logger.debug("FORM FIELDS: " + json.dumps(log_items, ensure_ascii=False))

    property_type = (str(form.get("property_type")) if form.get("property_type") else "apartment").strip().lower()

    canonical_override: Optional[List[str]] = None
    raw = form.get("canonical_order")
    if raw:
        try:
            s = str(raw)
            if s.strip().startswith("["):
                canonical_override = json.loads(s)
            else:
                canonical_override = [x.strip() for x in s.split(",") if x.strip()]
        except Exception as e:
            logger.warning(f"No pude parsear canonical_order: {e}")

    image_parts = collect_images_from_form(form)
    logger.debug(f"IMAGES FOUND: {len(image_parts)}")
    if not image_parts:
        raise HTTPException(400, "Envía al menos una imagen (img1..img50) como archivo.")

    canonical = get_canonical(property_type, canonical_override)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(property_type, canonical, image_parts)}
    ]

    # ─── OpenAI (gpt-4o-mini usa max_tokens) ───
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=2000,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "ImageOrdering", "schema": model_schema(), "strict": True}
            }
        )
    except BadRequestError as e:
        logger.exception(f"OpenAI BadRequest: {e}")
        raise HTTPException(502, f"Error llamando a OpenAI: {e}")
    except Exception as e:
        logger.exception(f"OpenAI error: {e}")
        raise HTTPException(502, f"Error llamando a OpenAI: {e}")

    # Parseo
    try:
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.exception(f"No pude parsear JSON del modelo: {e}")
        raise HTTPException(502, "Respuesta inválida del modelo")

    data.setdefault("version", "1.0")
    data["property_type"] = property_type
    data.setdefault("canonical_order", canonical)

    logger.info(f"DONE classify images={len(data.get('images', []))}")
    return JSONResponse(content=data)
