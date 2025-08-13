# --- NUEVO: build_messages SIN start_idx (usa labels legibles) ---
def build_messages(image_parts, labels=None):
    user_content = []
    for i, part in enumerate(image_parts, start=1):
        tag = (labels[i-1] if labels and i-1 < len(labels) and labels[i-1] else f"img{i}").strip()
        user_content.append({"type": "text", "text": f"Imagen {tag}:"})
        user_content.append(part)
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",   "content": user_content},
    ]

# --- NUEVO: normaliza y usa labels/IDs provistos o filenames ---
def normalize_and_fill(block_data: dict, labels: list, NIVEL_MAP=None, LOWER_MAP=None):
    resultados = []
    if isinstance(block_data, dict):
        resultados = block_data.get("resultados", [])
    out = []
    count = len(labels)
    for i in range(count):
        rec = resultados[i] if i < len(resultados) and isinstance(resultados[i], dict) else {}
        amb_raw = (rec.get("ambiente") or "").strip()
        amb_key = amb_raw.lower()
        if LOWER_MAP and amb_key in LOWER_MAP:
            amb = LOWER_MAP[amb_key]
        else:
            amb = "Otro"
        nivel = NIVEL_MAP.get(amb, 1) if NIVEL_MAP else 1
        et = rec.get("etiquetas", [])
        if not isinstance(et, list):
            et = []
        et = [str(x).strip() for x in et if str(x).strip()][:4]
        out.append({
            "imagen_id": labels[i],  # <- el ID/filename que te devolvemos
            "nivel": int(nivel),
            "ambiente": amb,
            "etiquetas": et,
        })
    return out

# --- NUEVO ENDPOINT: hasta 5 imágenes, sin start_idx; usa ids o filenames ---
@app.post("/classify-5")
async def classify_5(
    img1: UploadFile = File(None), img2: UploadFile = File(None),
    img3: UploadFile = File(None), img4: UploadFile = File(None),
    img5: UploadFile = File(None),
    ids: str = "",          # opcional: "img1,img2,img3,img4,img5" (mismo orden que envías)
    listing_id: str = ""    # opcional: te lo devolvemos en meta para merge
):
    files = [img1, img2, img3, img4, img5]
    parts, names = [], []
    for f in files:
        part = to_image_part(f)
        if part:
            parts.append(part)
            names.append((f.filename or "").strip() or None)

    if not parts:
        raise HTTPException(400, "Envía al menos una imagen válida (jpg/png).")
    if len(parts) > 5:
        raise HTTPException(400, "Máximo 5 imágenes por request.")

    # Labels/IDs para devolver: prioridad ids (query), si no filenames, si no img1..imgN
    provided_ids = [x.strip() for x in ids.split(",")] if ids else []
    labels = []
    for i in range(len(parts)):
        if i < len(provided_ids) and provided_ids[i]:
            labels.append(provided_ids[i])
        elif names[i]:
            labels.append(names[i])
        else:
            labels.append(f"img{i+1}")

    # Tokens de salida conservadores
    per_image = 34
    base_overhead = 220
    max_tokens = min(base_overhead + per_image * len(parts), 1200)

    messages = build_messages(parts, labels=labels)

    try:
        resp = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        logger.debug(f"RAW OpenAI: {content}")
        data = json.loads(content)
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        raise HTTPException(502, f"Error OpenAI: {e}")

    resultados = normalize_and_fill(
        data, labels=labels, NIVEL_MAP=NIVEL_MAP, LOWER_MAP=LOWER_MAP
    )

    out = {
        "resultados": resultados,
        "meta": {
            "listing_id": listing_id,
            "count": len(parts),
            "max_tokens": max_tokens,
        }
    }
    logger.info(f"classify-5 OK | {len(resultados)} clasificadas | listing_id={listing_id or '-'}")
    return JSONResponse(content=out)
