import os, json, base64, time, logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import openai, dotenv

# ── Credenciales ──────────────────────────────────────────────────────────────
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini-2024-07-18")

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("ia-classifier")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(title="Image Room Classifier (5 imágenes)")

# ── Lista CANÓNICA de ambientes (mismo orden que en el prompt) ───────────────
AMBIENTES_ORDER = [
  'Sala', 'Comedor', 'Cocina', 'Cocina integrada', 'Kitchenette',
  'Dormitorio', 'Dormitorio en suite',
  'Baño', 'Toilette',
  'Lavadero', 'Despensa', 'Baulera', 'Placard', 'Vestidor',
  'Home office', 'Oficina', 'Recepción', 'Hall de entrada', 'Pasillo',
  'Escalera', 'Sótano', 'Altillo',
  'Balcón', 'Terraza', 'Azotea', 'Roof garden', 'Galería',
  'Patio', 'Jardín', 'Quincho', 'Asador',
  'Cochera', 'Cochera subterránea', 'Estacionamiento', 'Estacionamiento visitantes',
  'Pileta', 'Solarium', 'Gimnasio', 'Sauna', 'Salón', 'Salón de eventos',
  'Cowork', 'Sala de juegos', 'Juegos infantiles', 'Laundry', 'Parrilla', 'Parque canino',
  'Fachada', 'Vista calle', 'Contrafrente',
  'Plano', 'Render', 'Maqueta',
  'Planta libre', 'Privado', 'Sala de reuniones', 'Auditorio', 'Archivo',
  'Data center', 'Sala de servidores', 'Comedor de personal', 'Cocina office', 'Baños públicos',
  'Lote', 'Terreno', 'Portón', 'Alambrado perimetral', 'Camino interno',
  'Casa principal', 'Casa de caseros', 'Casa de huéspedes',
  'Galpón', 'Depósito', 'Taller', 'Corrales', 'Manga', 'Caballerizas',
  'Silo', 'Tanque de agua', 'Aguadas', 'Pozo de agua', 'Arroyo', 'Río', 'Laguna', 'Monte', 'Arboleda', 'Pastura', 'Cultivo',
  'Club house', 'Garita de acceso', 'Seguridad', 'Circuito cerrado (CCTV)',
  'Calles internas', 'Bicicletero', 'Cocheras de cortesía', 'Plaza central', 'Parque', 'Senderos',
  'Cancha de tenis', 'Cancha de pádel', 'Cancha de fútbol', 'Multicancha', 'Laguna artificial',
  'Local comercial', 'Isla comercial', 'Vidriera', 'Pasillo comercial', 'Hall central',
  'Patio de comidas', 'Restaurante', 'Cafetería', 'Back office', 'Gerencia',
  'Área de carga y descarga', 'Montacargas', 'Escalera mecánica', 'Ascensores', 'Terraza técnica', 'Cartelería', 'Tótem',
  'Sala de máquinas', 'Tablero eléctrico', 'Grupo electrógeno',
  'Otro'
]
N = len(AMBIENTES_ORDER)
NIVEL_MAP = {amb: (N - idx) for idx, amb in enumerate(AMBIENTES_ORDER)}  # primero=N, último=1
LOWER_MAP = {amb.lower(): amb for amb in AMBIENTES_ORDER}

# ── Devuelve dict si es imagen válida, o None si viene vacío/no imagen ───────
def to_image_part(f: UploadFile):
    if not f:
        return None
    ct = (f.content_type or "").lower()
    try:
        data = f.file.read()
    except Exception:
        return None
    if not data:
        return None
    if not ct.startswith("image/"):
        return None
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{ct};base64,{b64}"}}

# ── Prompt del clasificador (simple) ─────────────────────────────────────────
SYSTEM_MSG = f"""
Eres un asistente que CLASIFICA fotos de propiedades inmobiliarias.

DEVUELVE EXCLUSIVAMENTE un JSON (sin texto extra) con esta estructura EXACTA:
{{
  "resultados": [
    {{
      "imagen_id": "img1",
      "nivel": <entero 1-{N}>,
      "ambiente": "<una de las opciones permitidas con la primera letra en mayuscula>",
      "etiquetas": ["<etiqueta1>", "<etiqueta2>"]
    }}
  ]
}}

Reglas:
- "imagen_id" debe ser "imgN" (img1, img2, ...) respetando el orden de entrada.
- "nivel": entero de 1 a {N}, calculado por posición en la lista de ambientes (el primero vale {N} y el último vale 1).
- "ambiente": elige exactamente UNA de estas opciones, devuelve con la primera letra en mayúscula (usa 'Otro' si dudas):
{AMBIENTES_ORDER}
- "etiquetas": lista corta (1 a 4) con palabras sencillas, por ejemplo:
  ['iluminada', 'amplia', 'moderna', 'renovada', 'equipada', 'ordenada',
   'con vista', 'ventilada', 'integrada', 'con isla', 'suite', 'placard',
   'semi-cubierta', 'a estrenar', 'antigua', 'rústica']
- No devuelvas imágenes/base64 ni texto fuera del JSON.
"""

def build_messages(image_parts):
    mapping_text = (
        f"Estas {len(image_parts)} imágenes corresponden a los IDs "
        f"img1..img{len(image_parts)} en ese orden. Devuelve 'resultados' en ese mismo orden."
    )
    user_content = [{"type": "text", "text": mapping_text}, *image_parts]
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user_content},
    ]

def normalize_and_fill(block_data, count: int):
    out = []
    resultados = block_data.get("resultados", []) if isinstance(block_data, dict) else []
    for i in range(count):
        rec = resultados[i] if i < len(resultados) and isinstance(resultados[i], dict) else {}
        amb_raw = (rec.get("ambiente") or "").strip()
        amb = LOWER_MAP.get(amb_raw.lower(), "Otro")
        nivel = NIVEL_MAP.get(amb, 1)
        et = rec.get("etiquetas", [])
        if not isinstance(et, list):
            et = []
        etiquetas = []
        for x in et[:4]:
            s = str(x).strip()
            if s:
                etiquetas.append(s)
        out.append({
            "imagen_id": f"img{i+1}",
            "nivel": int(nivel),
            "ambiente": amb,
            "etiquetas": etiquetas,
        })
    return out

# ── Endpoint: EXACTO 1..5 imágenes, una sola llamada al modelo ──────────────
@app.post("/classify-5")
@app.post("/classify-simple")   # por compatibilidad si tu front ya lo usa
@app.post("/extract-image")     # por compatibilidad con el nombre viejo
async def classify_5(
    img1: UploadFile = File(None),
    img2: UploadFile = File(None),
    img3: UploadFile = File(None),
    img4: UploadFile = File(None),
    img5: UploadFile = File(None),
):
    parts = [to_image_part(img1), to_image_part(img2), to_image_part(img3),
             to_image_part(img4), to_image_part(img5)]
    image_parts = [p for p in parts if p is not None]

    if not image_parts:
        raise HTTPException(400, "Envía al menos una imagen válida (jpg/png).")
    if len(image_parts) > 5:
        raise HTTPException(400, "Máximo 5 imágenes por request.")

    # Tokens de salida conservadores para 1..5 filas
    per_image = 34
    base_overhead = 220
    max_tokens = min(base_overhead + per_image * len(image_parts), 1200)

    messages = build_messages(image_parts)

    try:
        t0 = time.perf_counter()
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        logger.debug(f"RAW OpenAI response: {content}")  # <- LOG del JSON crudo
        data = json.loads(content)
        elapsed = time.perf_counter() - t0
        logger.info(f"classify-5 OK | imgs={len(image_parts)} | t={elapsed:.2f}s | max_tokens={max_tokens}")
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise HTTPException(502, f"Error OpenAI: {e}")

    resultados = normalize_and_fill(data, count=len(image_parts))

    out = {
        "resultados": resultados,
        "meta": {
            "count": len(image_parts),
            "max_tokens": max_tokens
        }
    }
    return JSONResponse(content=out)
