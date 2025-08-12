import os, json, base64, logging, sys, time
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import openai, dotenv

# ── Logging para Railway ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ia-classifier")

# ── Credenciales ──────────────────────────────────────────────────────────────
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini-2024-07-18")

app = FastAPI(title="Image Room Classifier (1-50 imágenes)")

@app.on_event("startup")
def _on_startup():
    logger.info(f"App iniciada | MODEL={MODEL}")

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
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{ct};base64,{b64}"}
    }

# ── Prompt del clasificador ───────────────────────────────────────────────────
SYSTEM_MSG = """
Eres un asistente que CLASIFICA fotos de propiedades inmobiliarias.

DEVUELVE EXCLUSIVAMENTE un JSON (sin texto extra) con esta estructura EXACTA:
{
  "resultados": [
    {
      "imagen_id": "img1",
      "nivel": <entero 1-124>,
      "ambiente": "<una de las opciones permitidas con la primera letra en mayuscula>",
      "etiquetas": ["<etiqueta1>", "<etiqueta2>"]
    }
  ]
}

Reglas:
- "imagen_id" debe ser "imgN" (img1, img2, ...) respetando el orden de entrada.
- "nivel" entero de 1 a 124 calculado por posición en la lista de ambientes (el primero vale 124 y el último vale 1); por ejemplo, 'sala' = 124 y 'otro' = 1.
- "ambiente": elige exactamente UNA de estas opciones, pero devuelvela con la primera letra en mayuscula:
 [
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
- "etiquetas": lista corta (1 a 4) con palabras sencillas, por ejemplo:
  ['iluminada', 'amplia', 'moderna', 'renovada', 'equipada', 'ordenada',
   'con vista', 'ventilada', 'integrada', 'con isla', 'suite', 'placard',
   'semi-cubierta', 'a estrenar', 'antigua', 'rústica']
- Si no estás seguro del ambiente, usa 'otro'.
- No inventes texto ni resúmenes; NO devuelvas imágenes ni base64.
- Solo JSON válido.
"""

def build_messages(image_parts):
    # user.content debe ser la lista de imágenes para visión
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": image_parts},
    ]

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
    t0 = time.perf_counter()

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
    image_parts = [p for p in raw_parts if p is not None]
    logger.info(f"Imágenes válidas encontradas: {len(image_parts)}")

    if not image_parts:
        raise HTTPException(400, "Envía al menos una imagen válida (jpg/png).")

    messages = build_messages(image_parts)

    # ── Cálculo dinámico de tokens de salida ──────────────────────────────────
    per_image = 18      # estimado por fila (id + nivel + ambiente + etiquetas)
    base_overhead = 400 # cabecera y estructura JSON
    max_toks = min(base_overhead + per_image * len(image_parts), 3500)
    logger.info(f"Llamando a OpenAI (modelo: {MODEL}, imgs: {len(image_parts)}, max_tokens: {max_toks})")

    try:
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=max_toks,                  # <- ajustado dinámicamente
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        data = json.loads(content)  # debe ser JSON válido

        # Métricas / uso
        elapsed = time.perf_counter() - t0
        usage = getattr(resp, "usage", None)
        prompt_toks = completion_toks = total_toks = None
        try:
            if usage:
                # objeto o dict, manejamos ambos
                prompt_toks = getattr(usage, "prompt_tokens", None) or (usage.get("prompt_tokens") if isinstance(usage, dict) else None)
                completion_toks = getattr(usage, "completion_tokens", None) or (usage.get("completion_tokens") if isinstance(usage, dict) else None)
                total_toks = getattr(usage, "total_tokens", None) or (usage.get("total_tokens") if isinstance(usage, dict) else None)
        except Exception:
            pass

        logger.info(
            f"OpenAI OK | resultados={len(data.get('resultados', []))} | "
            f"tiempo={elapsed:.2f}s | tokens p={prompt_toks} c={completion_toks} t={total_toks}"
        )

    except Exception as e:
        logger.exception("OpenAI error")
        # Devuelve 500 con el texto del error para depurar rápidamente
        raise HTTPException(500, f"Error OpenAI: {e}")

    # Opcional: agregar conteo
    data.setdefault("meta", {})
    data["meta"]["total_recibidas"] = len(raw_parts)
    data["meta"]["total_clasificadas"] = len(image_parts)
    data["meta"]["max_tokens_usados"] = max_toks

    return JSONResponse(content=data)

# (Opcional para ejecutar local)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
