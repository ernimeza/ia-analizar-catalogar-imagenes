import os, json, base64, logging, time, math, asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import openai, dotenv

# ── Credenciales ──────────────────────────────────────────────────────────────
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini-2024-07-18")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ia-classifier")

app = FastAPI(title="Image Room Classifier (1-50 imágenes)")

# ── Orden maestro de ambientes (importancia) ──────────────────────────────────
AMBIENTES = [
    # Interiores
    'Sala','Dormitorio','Dormitorio en suite','Cocina','Cocina integrada','Kitchenette','Comedor',
    'Baño','Toilette','Vestidor','Placard','Lavadero','Despensa','Baulera',
    'Oficina','Home office','Recepción','Hall de entrada','Pasillo','Escalera','Altillo','Sótano',
    # Amenidades
    'Pileta','Quincho','Asador','Parrilla','Salón','Salón de eventos','Gimnasio','Sauna',
    'Sala de juegos','Juegos infantiles','Cowork','Laundry','Solarium','Club house','Seguridad',
    'Circuito cerrado (CCTV)','Garita de acceso','Plaza central','Parque','Senderos',
    'Cancha de fútbol','Cancha de pádel','Cancha de tenis','Multicancha','Parque canino',
    'Cocheras de cortesía','Calles internas','Bicicletero','Laguna artificial',
    # Exteriores / cocheras / vistas
    'Fachada','Vista calle','Contrafrente','Patio','Jardín','Balcón','Terraza','Azotea','Galería','Roof garden',
    'Cochera','Cochera subterránea','Estacionamiento','Estacionamiento visitantes',
    # Terreno / campo
    'Lote','Terreno','Casa principal','Casa de caseros','Casa de huéspedes','Portón','Alambrado perimetral',
    'Camino interno','Tanque de agua','Pozo de agua','Aguadas','Laguna','Arroyo','Río','Monte','Arboleda',
    'Pastura','Cultivo',
    # Técnicos / renders / planos
    'Plano','Render','Maqueta','Planta libre','Privado','Sala de reuniones','Auditorio','Archivo',
    'Data center','Sala de servidores','Comedor de personal','Cocina office','Baños públicos',
    # Comercial / shopping
    'Local comercial','Isla comercial','Vidriera','Pasillo comercial','Hall central','Patio de comidas',
    'Restaurante','Cafetería','Back office','Gerencia','Área de carga y descarga','Montacargas',
    'Escalera mecánica','Ascensores','Terraza técnica','Cartelería','Tótem','Sala de máquinas',
    'Tablero eléctrico','Grupo electrógeno',
    # Industrial
    'Galpón','Depósito','Taller','Corrales','Manga','Caballerizas','Silo',
    # Catch-all
    'Otro'
]
NIVEL_MAP = {name: len(AMBIENTES) - i for i, name in enumerate(AMBIENTES)}  # 124..1

# ── Utilidad: UploadFile -> image_url (data:) ─────────────────────────────────
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

# ── Prompt por bloque, indicando rango exacto de imgN ─────────────────────────
def build_system_msg(start_idx: int, count: int):
    opciones = ", ".join(f"'{a}'" for a in AMBIENTES)
    end_idx = start_idx + count - 1
    return f"""
Eres un asistente que CLASIFICA fotos de propiedades inmobiliarias.

DEVUELVE EXCLUSIVAMENTE un JSON (sin texto extra) con esta estructura EXACTA:
{{
  "resultados": [
    {{
      "imagen_id": "img{start_idx}",
      "ambiente": "<una de las opciones permitidas>",
      "etiquetas": ["<etiqueta1>", "<etiqueta2>"]
    }}
  ]
}}

Reglas rígidas para este bloque:
- Debes devolver **exactamente {count} objetos** en "resultados".
- Los "imagen_id" deben ser, en este orden, desde "img{start_idx}" hasta "img{end_idx}" (uno por cada imagen recibida).
- "ambiente": elige exactamente UNA de: [ {opciones} ].
- Si no estás seguro, usa 'Otro' y deja etiquetas [].
- No calcules niveles; el servidor los definirá.
- Prohibido devolver texto fuera del JSON. Solo JSON válido.
""".strip()

def build_messages(image_parts, start_idx):
    return [
        {"role": "system", "content": build_system_msg(start_idx, len(image_parts))},
        {"role": "user",   "content": image_parts},
    ]

# ── Llamada a OpenAI por bloque con reintentos ───────────────────────────────
async def classify_block(image_parts, start_idx, per_image_tokens=40, base_overhead=200, retries=2):
    max_toks = min(base_overhead + per_image_tokens * len(image_parts), 1200)
    messages = build_messages(image_parts, start_idx)
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
            return data
        except Exception as e:
            logger.warning(f"Chunk {start_idx} intento {attempt+1} fallo: {e}")
            if attempt < retries:
                await asyncio.sleep(1.5 * (attempt + 1))
            else:
                raise

# ── Normaliza salida del bloque y garantiza cobertura ─────────────────────────
def normalize_and_fill(block_data, start_idx, count):
    # Esperados: img{start}..img{end}
    end_idx = start_idx + count - 1
    expected_ids = [f"img{i}" for i in range(start_idx, end_idx + 1)]

    resultados = block_data.get("resultados") or []
    by_id = {str(r.get("imagen_id")): r for r in resultados if r and r.get("imagen_id")}

    final = []
    for img_id in expected_ids:
        item = by_id.get(img_id)
        if not item:
            item = {"imagen_id": img_id, "ambiente": "Otro", "etiquetas": []}
        # Normalizar ambiente y calcular nivel en backend
        amb = str(item.get("ambiente", "Otro")).strip() or "Otro"
        # corregimos capitalización si hiciera falta
        amb_fixed = next((a for a in AMBIENTES if a.lower() == amb.lower()), "Otro")
        item["ambiente"] = amb_fixed
        item["nivel"] = NIVEL_MAP.get(amb_fixed, 1)
        # Asegurar etiquetas lista
        et = item.get("etiquetas")
        if not isinstance(et, list):
            item["etiquetas"] = []
        final.append(item)
    return final

# ── Endpoint principal (acepta hasta 50 archivos) ─────────────────────────────
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
    logger.info(f"Imágenes válidas: {total}")

    # ── Procesar en bloques ───────────────────────────────────────────────────
    CHUNK_SIZE = 10
    tasks = []
    start_index = 1
    for i in range(0, total, CHUNK_SIZE):
        chunk = image_parts_all[i:i+CHUNK_SIZE]
        tasks.append(classify_block(chunk, start_index + i))
    # Ejecutar secuencialmente (más simple y seguro con rate limits)
    resultados_comb = []
    for coro in tasks:
        block_data = await coro
        fixed = normalize_and_fill(
            block_data,
            start_idx=1 + len(resultados_comb),
            count=len(json.loads(json.dumps(block_data.get("resultados") or []) ))  # fallback, será reparado abajo
        )
        # Ojo: el fill necesita usar el rango correcto del bloque:
        # Recalcular con base al tamaño real del chunk y el índice global
    resultados_comb = []  # rehacemos correctamente
    # Segunda pasada correcta con los rangos reales
    resultados_finales = []
    for i in range(0, total, CHUNK_SIZE):
        chunk = image_parts_all[i:i+CHUNK_SIZE]
        start_idx = 1 + i
        block_data = await classify_block(chunk, start_idx)
        fixed = normalize_and_fill(block_data, start_idx=start_idx, count=len(chunk))
        resultados_finales.extend(fixed)

    data = {"resultados": resultados_finales}

    # Meta informativa
    data.setdefault("meta", {})
    data["meta"]["total_recibidas"] = len(raw_parts)
    data["meta"]["total_clasificadas"] = total
    data["meta"]["chunks"] = math.ceil(total / CHUNK_SIZE)

    logger.info("JSON IA (merge):\n" + json.dumps(data, ensure_ascii=False, indent=2))
    return JSONResponse(content=data)
