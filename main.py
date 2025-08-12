import os, json, base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import openai, dotenv

# ── Credenciales ──────────────────────────────────────────────────────────────
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini-2024-07-18")

app = FastAPI(title="Image Room Classifier (1-50 imágenes)")

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
      "ambiente": "<una de las opciones permitidas>",
      "etiquetas": ["<etiqueta1>", "<etiqueta2>"]
    }
  ]
}

Reglas:
- "imagen_id" debe ser "imgN" (img1, img2, ...) respetando el orden de entrada.
- "Nivel" entero de 1 a 124 calculado por posición en la lista de ambientes (el primero vale 124 y el último vale 1); por ejemplo, 'sala' = 124 y 'otro' = 1.
- "ambiente": elige exactamente UNA de:
  [
  'sala', 'comedor', 'cocina', 'cocina integrada', 'kitchenette',
  'dormitorio', 'dormitorio en suite',
  'baño', 'toilette',
  'lavadero', 'despensa', 'baulera', 'placard', 'vestidor',
  'home office', 'oficina', 'recepción', 'hall de entrada', 'pasillo',
  'escalera', 'sótano', 'altillo',
  'balcón', 'terraza', 'azotea', 'roof garden', 'galería',
  'patio', 'jardín', 'quincho', 'asador',
  'cochera', 'cochera subterránea', 'estacionamiento', 'estacionamiento visitantes',
  'pileta', 'solarium', 'gimnasio', 'sauna', 'salón', 'salón de eventos',
  'cowork', 'sala de juegos', 'juegos infantiles', 'laundry', 'parrilla', 'parque canino',
  'fachada', 'vista calle', 'contrafrente',
  'plano', 'render', 'maqueta',
  'planta libre', 'privado', 'sala de reuniones', 'auditorio', 'archivo',
  'data center', 'sala de servidores', 'comedor de personal', 'cocina office', 'baños públicos',
  'lote', 'terreno', 'portón', 'alambrado perimetral', 'camino interno',
  'casa principal', 'casa de caseros', 'casa de huéspedes',
  'galpón', 'depósito', 'taller', 'corrales', 'manga', 'caballerizas',
  'silo', 'tanque de agua', 'aguadas', 'pozo de agua', 'arroyo', 'río', 'laguna', 'monte', 'arboleda', 'pastura', 'cultivo',
  'club house', 'garita de acceso', 'seguridad', 'circuito cerrado (CCTV)',
  'calles internas', 'bicicletero', 'cocheras de cortesía', 'plaza central', 'parque', 'senderos',
  'cancha de tenis', 'cancha de pádel', 'cancha de fútbol', 'multicancha', 'laguna artificial',
  'local comercial', 'isla comercial', 'vidriera', 'pasillo comercial', 'hall central',
  'patio de comidas', 'restaurante', 'cafetería', 'back office', 'gerencia',
  'área de carga y descarga', 'montacargas', 'escalera mecánica', 'ascensores', 'terraza técnica', 'cartelería', 'tótem',
  'sala de máquinas', 'tablero eléctrico', 'grupo electrógeno',
  'otro'
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

    if not image_parts:
        raise HTTPException(400, "Envía al menos una imagen válida (jpg/png).")

    messages = build_messages(image_parts)

    try:
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=1000,  # suficiente para 50 filas cortas
            response_format={"type": "json_object"},  # JSON garantizado
        )
        content = resp.choices[0].message.content
        data = json.loads(content)  # debe ser JSON válido
    except Exception as e:
        # Devuelve 500 con el texto del error para depurar rápidamente
        raise HTTPException(500, f"Error OpenAI: {e}")

    # Opcional: agregar conteo
    data.setdefault("meta", {})
    data["meta"]["total_recibidas"] = len(raw_parts)
    data["meta"]["total_clasificadas"] = len(image_parts)

    return JSONResponse(content=data)
