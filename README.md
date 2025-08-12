# ia-image-sorter

Microservicio FastAPI que clasifica y ordena fotos inmobiliarias por ambiente y devuelve JSON estructurado para Bubble.

## Variables de entorno
- `OPENAI_API_KEY` (obligatoria)
- `MODEL` (opcional): `gpt-5-mini` (recomendado) o `gpt-4o-mini-2024-07-18`

## Levantar local
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
uvicorn main:app --reload
