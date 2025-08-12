import os, json, base64, logging, re
from typing import List, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import dotenv, openai

# ──────────────────────────── Config ────────────────────────────
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini-2024-07-18")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ia-classifier")

app = FastAPI(title="Image Room Classifier (simple)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────── Helpers ──────────────────────────────
ALLOWED_AMBIENTES = [
    "sala", "comedor", "cocina", "habitacion", "baño",
    "fachada", "exterior", "jardin", "patio",
    "balcon", "terraza", "cochera", "garaje",
    "pasillo", "hall", "lavadero", "quincho", "piscina",
    "plano", "texto", "otros"
]

SUGERIDAS_ETIQUETAS = [
    "iluminada", "amplia", "moderna", "clasica", "remodelada",
    "minimalista", "con vista", "con muebles", "sin muebles",
    "piso de madera", "piso ceramico", "aire acondicionado",
    "ventanales", "integrada", "isla", "placard", "en suite",
    "ducha", "bañera", "doble bacha", "parrilla", "verde",
    "cubierto", "descubierto", "techado", "en construccion",
    "a refaccionar"
]

# Schema estricto para obligar JSON válido
JSON_SCHEMA = {
    "name": "ImagenTags",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "resultados": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "imagen_id": {"type": "string"},
                        "ambiente":  {"type": "string", "enum": ALLOWED_AMBIENTES},
                        "etiquetas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 0,
                            "maxItems": 5
                        }
                    },
                    "required": ["imagen_id", "ambiente", "etiquetas"]
                }
            }
        },
        "required": ["resultados"]
    },
    "strict": True
}

def file_to_image_part(upload, image_id: str):
    if upload is None:
        return None, False
    try:
        ct = (upload.content_type or "").lower()
        data = upload.file.read()
    except Exception:
        return None, False
    if not data or not ct.startswith("image/"):
        return None, False

    b64 = base64.b64encode(data).decode()
    part = {
        "type": "image_url",
        "image_url": {"url": f"data:{ct};base64,{b64}", "detail": "low"}
    }
    return part, True

def collect_images_from_form(form, max_n: int = 50) -> Tuple[List[dict], List[str]]:
    parts, ids = [], []
    for i in range(1, max_n + 1):
        key = f"img{i}"
        upload = form.get(key)
        if upload is None:
            continue
        part, ok = file_to_image_part(upload, key)
        if ok:
            parts.append(part)
            ids.append(key)
    logger.info(f"Imágenes válidas encontradas: {len(parts)}")
    return parts, ids

def best_effort_json(text: str):
    """Intenta parsear; si falla, intenta recortar al primer '{' y último '}'."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m1 = text.find("{")
    m2 = text.rfind("}")
    if m1 != -1 and m2 != -1 and m2 > m1:
        snippet = text[m1:m2+1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    raise

# ───────────────────────── Endpoints ────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/classify-simple")
async def classify_simple(request: Request):
    """
    Recibe img1..img50 (opcionales) como archivos y devuelve:
    {
      "resultados": [
        {"imagen_id": "img1", "ambiente": "sala", "etiquetas": ["iluminada","amplia"]},
        ...
      ]
    }
    """
    form = await request.form()
    image_parts, image_ids = collect_images_from_form(form, max_n=50)

    if not image_parts:
        raise HTTPException(400, "Envía al menos una imagen válida (jpg/png).")

    system_prompt = f"""
Eres un asistente que clasifica fotos inmobiliarias.
Para CADA imagen, devuelve:
- "ambiente": uno de {ALLOWED_AMBIENTES}.
- "etiquetas": 2-5 descriptores en minúsculas, simples. Ejemplos: {SUGERIDAS_ETIQUETAS}.
Reglas:
- Mantén el MISMO ORDEN de entrada.
- Responde SOLO el JSON requerido.
"""

    user_intro = (
        "Analiza y etiqueta estas imágenes en el orden dado. "
        "Devuelve exactamente un elemento por imagen."
    )

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": [{"type": "text", "text": user_intro}] + image_parts},
    ]

    try:
        logger.info("Llamando a OpenAI (modelo: %s, imgs: %d)", MODEL, len(image_parts))
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=800,
            response_format={"type": "json_schema", **JSON_SCHEMA},
        )
        content = resp.choices[0].message.content
        data = best_effort_json(content)  # robustez extra
    except Exception as e:
        logger.exception("OpenAI error: %s", e)
        raise HTTPException(502, f"Error llamando a OpenAI: {e}")

    resultados = data.get("resultados", [])
    if not isinstance(resultados, list):
        resultados = []

    # Normalizamos longitud y campos
    if len(resultados) < len(image_ids):
        for i in range(len(image_ids) - len(resultados)):
            resultados.append({
                "imagen_id": image_ids[len(resultados)],
                "ambiente": "otros",
                "etiquetas": []
            })
    resultados = resultados[: len(image_ids)]

    for idx, item in enumerate(resultados):
        item["imagen_id"] = image_ids[idx]
        amb = (item.get("ambiente") or "otros").lower()
        if amb not in ALLOWED_AMBIENTES:
            amb = "otros"
        et = item.get("etiquetas")
        if not isinstance(et, list):
            et = []
        item["ambiente"] = amb
        item["etiquetas"] = [str(x).lower()[:60] for x in et][:5]

    out = {"resultados": resultados}
    logger.info("Clasificación OK. Imágenes: %d", len(resultados))
    return JSONResponse(out)
