import os, json, base64, logging
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
    # descriptores genéricos y frecuentes
    "iluminada", "amplia", "moderna", "clasica", "remodelada",
    "minimalista", "con vista", "con muebles", "sin muebles",
    "piso de madera", "piso ceramico", "aire acondicionado",
    "ventanales", "integrada", "isla", "placard", "en suite",
    "ducha", "bañera", "doble bacha", "parrilla", "verde",
    "cubierto", "descubierto", "techado", "en construccion",
    "a refaccionar"
]

def file_to_image_part(upload, image_id: str):
    """
    Convierte UploadFile a parte de mensaje (image_url data:)
    Devuelve (parte, ok:bool). Si no es imagen o está vacío, ok=False.
    """
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
        "image_url": {"url": f"data:{ct};base64,{b64}", "detail": "low"}  # low = más rápido/barato
    }
    return part, True

def collect_images_from_form(form, max_n: int = 50) -> Tuple[List[dict], List[str]]:
    """
    Recolecta img1..imgN del form (si existen, como UploadFile).
    Devuelve (image_parts, ids_en_orden)
    """
    parts, ids = [], []
    count = 0
    for i in range(1, max_n + 1):
        key = f"img{i}"
        upload = form.get(key)
        if upload is None:
            continue
        part, ok = file_to_image_part(upload, key)
        if ok:
            parts.append(part)
            ids.append(key)
            count += 1
    logger.info(f"Imágenes válidas encontradas: {count}")
    return parts, ids

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

    # Instrucciones al modelo (en español, bien acotado)
    system_prompt = f"""
Eres un asistente que clasifica fotos inmobiliarias.
Para CADA imagen recibida, debes devolver un objeto con:
- "ambiente": uno de {ALLOWED_AMBIENTES}.
  Si dudas, usa "otros".
- "etiquetas": lista corta (2-5) de descriptores útiles (en minúsculas).
  Ejemplos frecuentes: {SUGERIDAS_ETIQUETAS}.
  Evita repetir palabras vacías; usa términos simples.

Reglas de salida IMPORTANTES:
- Devuelve SOLO un JSON con la forma:
  {{"resultados":[{{"imagen_id":"imgX","ambiente":"...","etiquetas":["..."]}}, ...]}}
- Mantén el MISMO ORDEN de las imágenes de entrada.
- No incluyas texto fuera del JSON. No devuelvas las imágenes ni URLs.
"""

    user_intro = (
        "Analiza y etiqueta estas imágenes en el orden dado. "
        "Devuelve exactamente un elemento por imagen en el mismo orden."
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
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
    except Exception as e:
        logger.exception("OpenAI error: %s", e)
        raise HTTPException(502, f"Error llamando a OpenAI: {e}")

    # Validación mínima y relleno si hiciera falta
    resultados = data.get("resultados", [])
    if not isinstance(resultados, list):
        resultados = []

    # Si el modelo devolvió menos items de los que enviamos, completamos con placeholders.
    if len(resultados) < len(image_ids):
        faltan = len(image_ids) - len(resultados)
        for i in range(faltan):
            resultados.append({
                "imagen_id": image_ids[len(resultados)],
                "ambiente": "otros",
                "etiquetas": []
            })
    # Si devolvió más, recortamos.
    resultados = resultados[: len(image_ids)]

    # Garantizamos que cada item tenga imagen_id correcto y campos esperados
    for idx, item in enumerate(resultados):
        item["imagen_id"] = image_ids[idx]
        item["ambiente"] = (item.get("ambiente") or "otros").lower()
        if item["ambiente"] not in ALLOWED_AMBIENTES:
            item["ambiente"] = "otros"
        et = item.get("etiquetas")
        if not isinstance(et, list):
            et = []
        # normalizamos strings
        item["etiquetas"] = [str(x).lower()[:60] for x in et][:5]

    out = {"resultados": resultados}
    logger.info("Clasificación OK. Imágenes: %d", len(resultados))
    return JSONResponse(out)
