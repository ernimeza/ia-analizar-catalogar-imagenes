import os, json, base64, asyncio, time, logging, re
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

app = FastAPI(title="Image Room Classifier (1-50 imágenes)")

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

# ── Prompt del clasificador ───────────────────────────────────────────────────
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

def build_messages(image_parts, start_idx):
    # Inserta un texto de mapeo para que el modelo sepa qué imgN corresponde a cada imagen del chunk
    mapping_text = (
        f"Estas {len(image_parts)} imágenes corresponden a los IDs "
        f"img{start_idx}..img{start_idx+len(image_parts)-1} en ese orden. "
        f"Devuelve 'resultados' en ese mismo orden."
    )
    user_content = [{"type": "text", "text": mapping_text}, *image_parts]
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user_content},
    ]

# ── Normalización y relleno por bloque ────────────────────────────────────────
def normalize_and_fill(block_data, start_idx: int, count: int):
    out = []
    resultados = []
    try:
        if isinstance(block_data, dict):
            resultados = block_data.get("resultados", [])
        else:
            resultados = []
    except Exception:
        resultados = []

    # Buscar por posición (el modelo suele respetar el orden); si falta, usar {}.
    for i in range(count):
        rec = resultados[i] if i < len(resultados) and isinstance(resultados[i], dict) else {}

        # Ambiente normalizado
        amb_raw = (rec.get("ambiente") or "").strip()
        amb = LOWER_MAP.get(amb_raw.lower(), "Otro")

        # Nivel forzado por nuestro mapa
        nivel = NIVEL_MAP.get(amb, 1)

        # Etiquetas saneadas
        et = rec.get("etiquetas", [])
        if not isinstance(et, list):
            et = []
        etiquetas = []
        for x in et[:4]:
            try:
                s = str(x).strip()
                if s:
                    etiquetas.append(s)
            except Exception:
                continue

        out.append({
            "imagen_id": f"img{start_idx + i}",
            "nivel": int(nivel),
            "ambiente": amb,
            "etiquetas": etiquetas,
        })

    return out

# ── Llamada por bloque con manejo de rate limit ──────────────────────────────
async def classify_block(image_parts, start_idx, per_image_tokens=36, base_overhead=180, retries=4):
    max_toks = min(base_overhead + per_image_tokens * len(image_parts), 900)
    messages = build_messages(image_parts, start_idx)
    delay = 1.5

    for attempt in range(retries + 1):
        try:
            t0 = time.perf_counter()
            resp = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                max_tokens=max_toks,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            data = json.loads(content)
            elapsed = time.perf_counter() - t0
            logger.info(f"Chunk {start_idx}-{start_idx+len(image_parts)-1} OK | t={elapsed:.2f}s | max_tokens={max_toks}")
            logger.debug(f"RAW chunk {start_idx}: {content}")
            return data

        except openai.RateLimitError as e:
            m = re.search(r"in ([0-9]+(?:\.[0-9]+)?)s", str(e))
            wait = float(m.group(1)) if m else delay
            logger.warning(f"Rate limit 429 en chunk {start_idx}. Esperando {wait:.2f}s y reintentando (intento {attempt+1}/{retries})")
            await asyncio.sleep(wait)
            delay *= 1.6

        except Exception as e:
            logger.warning(f"Error en chunk {start_idx} (intento {attempt+1}/{retries}): {e}")
            if attempt < retries:
                await asyncio.sleep(delay)
                delay *= 1.5
            else:
                raise

# ── Endpoint: acepta img1..img50 como archivos (opcionales) ──────────────────
@app.post("/classify-simple")
@app.post("/extract-image")
async def classify_simple(
    img1: UploadFile = File(None),  img2: UploadFile = File(None),  img3: UploadFile = File(None),
    img4: UploadFile = File(None),  img5: UploadFile = File(None),  img6: UploadFile = File(None),
    img7: UploadFile = File(None),  img8: UploadFile = File(None),  img9: UploadFile = File(None),
    img10: UploadFile = File(None), img11: UploadFile = File(None), img12: UploadFile = File(None),
    img13: UploadFile = File(None), img14: UploadFile = File(None), img15: UploadFile = File(None),
    img16: UploadFile = File(None), img17: UploadFile = File(None), img18: UploadFile = File(None),
    img19: UploadFile = File(None), img20: UploadFile = File(None), img21: UploadFile = File(None),
    img22: UploadFile = File(None), img23: UploadFile = File(None), img24: UploadFile = File(None),
    img25: UploadFile = File(None), img26: UploadFile = File(None), img27: UploadFile = File(None),
    img28: UploadFile = File(None), img29: UploadFile = File(None), img30: UploadFile = File(None),
    img31: UploadFile = File(None), img32: UploadFile = File(None), img33: UploadFile = File(None),
    img34: UploadFile = File(None), img35: UploadFile = File(None), img36: UploadFile = File(None),
    img37: UploadFile = File(None), img38: UploadFile = File(None), img39: UploadFile = File(None),
    img40: UploadFile = File(None), img41: UploadFile = File(None), img42: UploadFile = File(None),
    img43: UploadFile = File(None), img44: UploadFile = File(None), img45: UploadFile = File(None),
    img46: UploadFile = File(None), img47: UploadFile = File(None), img48: UploadFile = File(None),
    img49: UploadFile = File(None), img50: UploadFile = File(None),
):
    # Construir lista solo con imágenes válidas (en orden)
    raw_parts = [
        to_image_part(img1),  to_image_part(img2),  to_image_part(img3),  to_image_part(img4),  to_image_part(img5),
        to_image_part(img6),  to_image_part(img7),  to_image_part(img8),  to_image_part(img9),  to_image_part(img10),
        to_image_part(img11), to_image_part(img12), to_image_part(img13), to_image_part(img14), to_image_part(img15),
        to_image_part(img16), to_image_part(img17), to_image_part(img18), to_image_part(img19), to_image_part(img20),
        to_image_part(img21), to_image_part(img22), to_image_part(img23), to_image_part(img24), to_image_part(img25),
        to_image_part(img26), to_image_part(img27), to_image_part(img28), to_image_part(img29), to_image_part(img30),
        to_image_part(img31), to_image_part(img32), to_image_part(img33), to_image_part(img34), to_image_part(img35),
        to_image_part(img36), to_image_part(img37), to_image_part(img38), to_image_part(img39), to_image_part(img40),
        to_image_part(img41), to_image_part(img42), to_image_part(img43), to_image_part(img44), to_image_part(img45),
        to_image_part(img46), to_image_part(img47), to_image_part(img48), to_image_part(img49), to_image_part(img50),
    ]
    image_parts_all = [p for p in raw_parts if p is not None]

    if not image_parts_all:
        raise HTTPException(400, "Envía al menos una imagen válida (jpg/png).")

    total = len(image_parts_all)
    logger.info(f"Imágenes válidas encontradas: {total}")

    # ── Procesar en bloques (secuencial) ──────────────────────────────────────
    CHUNK_SIZE = 10  # si ves 429, baja a 8 o 6
    resultados_finales = []

    for i in range(0, total, CHUNK_SIZE):
        chunk = image_parts_all[i:i+CHUNK_SIZE]
        start_idx = 1 + i
        block_data = await classify_block(chunk, start_idx)
        fixed = normalize_and_fill(block_data, start_idx=start_idx, count=len(chunk))
        resultados_finales.extend(fixed)

    data = {
        "resultados": resultados_finales,
        "meta": {
            "total_recibidas": len(raw_parts),
            "total_clasificadas": total,
            "chunk_size": CHUNK_SIZE,
        }
    }

    return JSONResponse(content=data)
