import os, json, base64, asyncio, time, logging, re, math
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

# ── Lista canónica y nivel ────────────────────────────────────────────────────
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

# Subconjunto de categorías comunes para el rescate (evitar “Otro”)
CORE_AMBIENTES = [
  'Sala','Dormitorio','Cocina','Baño','Comedor','Balcón','Terraza','Patio','Jardín',
  'Cochera','Estacionamiento','Pileta','Gimnasio','Sauna','Salón','Salón de eventos',
  'Oficina','Recepción','Hall de entrada','Pasillo','Quincho','Parrilla','Fachada',
  'Vista calle','Contrafrente','Plano','Render','Maqueta','Laundry','Vestidor','Placard'
]

# ── Util: imagen a parte vision ───────────────────────────────────────────────
def to_image_part(f: UploadFile):
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

# ── Mensajes por bloque (alineación por imagen) ──────────────────────────────
def build_messages_chunk(image_parts, start_idx, system_msg):
    # Marcamos cada imagen explícitamente: "Imagen imgN:"
    user_content = []
    for i, part in enumerate(image_parts, start=0):
        img_id = f"img{start_idx + i}"
        user_content.append({"type": "text", "text": f"Imagen {img_id}:"})
        user_content.append(part)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_content},
    ]

# ── Prompt principal ─────────────────────────────────────────────────────────
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
- Devuelve un elemento en "resultados" por CADA imagen recibida, en el MISMO orden.
- "imagen_id" debe ser "imgN" (img1, img2, ...), coincidiendo con las leyendas "Imagen imgN:" provistas.
- "ambiente": elige exactamente UNA de estas opciones (usa 'Otro' solo si ninguna aplica de verdad):
{AMBIENTES_ORDER}
- "nivel": entero de 1 a {N} según la posición del "ambiente" en la lista anterior (el primero vale {N} y el último vale 1).
- "etiquetas": 1 a 4 palabras sencillas (p. ej. 'iluminada','amplia','moderna','equipada','con vista','a estrenar').
- Evita usar 'Otro' salvo que no encaje en ninguna categoría. Si dudas, prefiere las categorías más comunes (Sala, Dormitorio, Cocina, Baño, Comedor).
- No devuelvas imágenes/base64 ni texto fuera del JSON.
"""

# ── Prompt de rescate (para "Otro") ──────────────────────────────────────────
RESCUE_SYSTEM = f"""
Clasifica ESTA única imagen en una de estas categorías COMUNES (evita 'Otro' salvo que sea imposible):
{CORE_AMBIENTES + ['Otro']}

Devuelve solo JSON:
{{"imagen_id":"<igual al pedido>","nivel":<1-{N}>,"ambiente":"<opción>","etiquetas":["..."]}}
"""

# ── Normalización y relleno ──────────────────────────────────────────────────
def normalize_and_fill(block_data, start_idx: int, count: int):
    out = []
    resultados = []
    if isinstance(block_data, dict):
        resultados = block_data.get("resultados", [])

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
            "imagen_id": f"img{start_idx + i}",
            "nivel": int(nivel),
            "ambiente": amb,
            "etiquetas": etiquetas,
        })
    return out

# ── Llamada con backoff ──────────────────────────────────────────────────────
async def call_openai(messages, max_tokens):
    delay = 1.5
    for attempt in range(5):
        try:
            return openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except openai.RateLimitError as e:
            m = re.search(r"in ([0-9]+(?:\.[0-9]+)?)s", str(e))
            wait = float(m.group(1)) if m else delay
            logger.warning(f"429 rate limit. Esperando {wait:.2f}s (intento {attempt+1}/5)")
            await asyncio.sleep(wait)
            delay *= 1.6
        except Exception as e:
            logger.warning(f"Error OpenAI: {e} (intento {attempt+1}/5)")
            await asyncio.sleep(delay)
            delay *= 1.4
    raise HTTPException(502, "No se pudo completar la clasificación (reintentos agotados).")

# ── Clasificación por bloque ─────────────────────────────────────────────────
async def classify_block(image_parts, start_idx, per_image_tokens=34, base_overhead=160):
    max_toks = min(base_overhead + per_image_tokens * len(image_parts), 900)
    messages = build_messages_chunk(image_parts, start_idx, SYSTEM_MSG)
    t0 = time.perf_counter()
    resp = await call_openai(messages, max_toks)
    content = resp.choices[0].message.content
    data = json.loads(content)
    elapsed = time.perf_counter() - t0
    logger.info(f"Chunk {start_idx}-{start_idx+len(image_parts)-1} OK | t={elapsed:.2f}s | max_tokens={max_toks}")
    logger.debug(f"RAW chunk {start_idx}: {content}")
    return data

# ── Rescate por imagen (si vino "Otro") ──────────────────────────────────────
async def rescue_single(image_part, imagen_id):
    messages = [
        {"role": "system", "content": RESCUE_SYSTEM},
        {"role": "user", "content": [
            {"type":"text","text": f"Clasifica evitando 'Otro' si es posible. imagen_id={imagen_id}."},
            image_part
        ]},
    ]
    resp = await call_openai(messages, max_tokens=220)
    data = json.loads(resp.choices[0].message.content)
    # Normalizamos igual
    amb_raw = (data.get("ambiente") or "").strip()
    amb = LOWER_MAP.get(amb_raw.lower(), "Otro")
    nivel = NIVEL_MAP.get(amb, 1)
    et = data.get("etiquetas", [])
    if not isinstance(et, list): et = []
    et = [str(x).strip() for x in et if str(x).strip()][:4]
    return {"imagen_id": imagen_id, "nivel": int(nivel), "ambiente": amb, "etiquetas": et}

# ── Endpoint ─────────────────────────────────────────────────────────────────
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

    # Parámetros
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "8"))  # baja si ves “Otro” repetido
    OTRO_RATIO_TRIGGER = float(os.getenv("OTRO_RATIO_TRIGGER", "0.25"))  # 25%
    USE_RESCUE = os.getenv("USE_RESCUE", "1") == "1"

    resultados_finales = []
    for i in range(0, total, CHUNK_SIZE):
        chunk = image_parts_all[i:i+CHUNK_SIZE]
        start_idx = 1 + i

        # 1) Clasificación normal del bloque
        block_data = await classify_block(chunk, start_idx)
        fixed = normalize_and_fill(block_data, start_idx=start_idx, count=len(chunk))

        # 2) Si hay muchos “Otro”, rescatar
        if USE_RESCUE:
            otros_idx = [j for j, rec in enumerate(fixed) if rec["ambiente"] == "Otro"]
            ratio = len(otros_idx) / len(fixed) if fixed else 0.0

            if ratio >= OTRO_RATIO_TRIGGER and len(otros_idx) > 0:
                logger.info(f"Rescate: {len(otros_idx)} de {len(fixed)} ({ratio:.0%}) en chunk {start_idx}")
                for j in otros_idx:
                    img_part = chunk[j]
                    img_id = fixed[j]["imagen_id"]
                    try:
                        rescued = await rescue_single(img_part, img_id)
                        fixed[j] = rescued
                    except Exception as e:
                        logger.warning(f"Rescate falló para {img_id}: {e}")

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
