import os, json, base64
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import openai, dotenv
from PIL import Image

# ── Credenciales ──────────────────────────────────────────────────────────────
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini-2024-07-18")

# Reducción de imagen (tuneable por env)
MAX_SIDE = int(os.getenv("MAX_SIDE", "1024"))       # px lado mayor
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70")) # 1..95

app = FastAPI(title="Image Room Classifier (1-50 imágenes)")

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_image_part(f: UploadFile):
    """Devuelve image_url vision con compresión/resize, o None si no es válida."""
    if not f:
        return None
    ct = (f.content_type or "").lower()
    if not ct.startswith("image/"):
        return None
    try:
        raw = f.file.read()
        if not raw:
            return None

        im = Image.open(BytesIO(raw))
        # Normaliza a RGB y aplana alfa
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")

        # Resize proporcional
        im.thumbnail((MAX_SIDE, MAX_SIDE))

        # JPEG liviano
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "low"  # velocidad > super detalle
            }
        }
    except Exception:
        return None

SYSTEM_MSG = """
Eres un asistente que CLASIFICA fotos de propiedades inmobiliarias.

DEVUELVE EXCLUSIVAMENTE un JSON con:
{
  "resultados": [
    { "imagen_id": "img1", "ambiente": "<opción>", "etiquetas": ["<tag1>", "<tag2>"] }
  ]
}

Reglas:
- "imagen_id" es "imgN" en el mismo orden de entrada.
- "ambiente": elegir UNA sola opción de la lista permitida.
- "etiquetas": 1 a 4 palabras simples (ej.: 'iluminada','amplia','moderna','a estrenar','rústica').
- Si no estás seguro del ambiente, usa 'otro'.
- Nada de texto adicional ni base64; solo JSON válido.
"""

# Lista de ambientes (en minúsculas; debe coincidir con enum del esquema)
AMBIENTES = [
    'sala','comedor','cocina','cocina integrada','kitchenette',
    'dormitorio','dormitorio en suite',
    'baño','toilette',
    'lavadero','despensa','baulera','placard','vestidor',
    'home office','oficina','recepción','hall de entrada','pasillo',
    'escalera','sótano','altillo',
    'balcón','terraza','azotea','roof garden','galería',
    'patio','jardín','quincho','asador',
    'cochera','cochera subterránea','estacionamiento','estacionamiento visitantes',
    'pileta','solarium','gimnasio','sauna','salón','salón de eventos',
    'cowork','sala de juegos','juegos infantiles','laundry','parrilla','parque canino',
    'fachada','vista calle','contrafrente',
    'plano','render','maqueta',
    'planta libre','privado','sala de reuniones','auditorio','archivo',
    'data center','sala de servidores','comedor de personal','cocina office','baños públicos',
    'lote','terreno','portón','alambrado perimetral','camino interno',
    'casa principal','casa de caseros','casa de huéspedes',
    'galpón','depósito','taller','corrales','manga','caballerizas',
    'silo','tanque de agua','aguadas','pozo de agua','arroyo','río','laguna','monte','arboleda','pastura','cultivo',
    'club house','garita de acceso','seguridad','circuito cerrado (CCTV)',
    'calles internas','bicicletero','cocheras de cortesía','plaza central','parque','senderos',
    'cancha de tenis','cancha de pádel','cancha de fútbol','multicancha','laguna artificial',
    'local comercial','isla comercial','vidriera','pasillo comercial','hall central',
    'patio de comidas','restaurante','cafetería','back office','gerencia',
    'área de carga y descarga','montacargas','escalera mecánica','ascensores','terraza técnica','cartelería','tótem',
    'sala de máquinas','tablero eléctrico','grupo electrógeno',
    'otro'
]

# JSON Schema estricto para obligar salida válida
JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "ImageClassificationResults",
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
                            "ambiente": {"type": "string", "enum": AMBIENTES},
                            "etiquetas": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 4
                            }
                        },
                        "required": ["imagen_id", "ambiente", "etiquetas"]
                    }
                },
                "meta": {
                    "type": "object",
                    "additionalProperties": True
                }
            },
            "required": ["resultados"]
        },
        "strict": True
    }
}

def build_messages(image_parts):
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": image_parts},
    ]

# ── Endpoint (img1..img50) ────────────────────────────────────────────────────
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
    image_parts = [p for p in raw_parts if p is not None]

    if not image_parts:
        raise HTTPException(400, "Envía al menos una imagen válida (jpg/png).")

    messages = build_messages(image_parts)

    try:
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=1000,
            response_format=JSON_SCHEMA,   # ⬅️ obliga JSON válido
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        raise HTTPException(500, f"Error OpenAI: {e}")

    # meta opcional
    data.setdefault("meta", {})
    data["meta"]["total_recibidas"] = len(raw_parts)
    data["meta"]["total_clasificadas"] = len(image_parts)

    return JSONResponse(content=data)
