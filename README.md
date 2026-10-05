# Unified Async LLM Client

Proyecto en Python 3.12 que implementa una interfaz común y asíncrona para cuatro proveedores:

- OpenAI
- Anthropic
- Groq
- Google Gemini

La misma aplicación puede cambiar de proveedor mediante `LLM_PROVIDER` sin modificar la lógica de `main.py`.

## 1. Requisitos

- Python 3.12
- Visual Studio Code
- Una API key de al menos uno de los proveedores (uso groq y gemini que son gratis)


## 2. Crear el proyecto desde cero en Windows

Abrí Visual Studio Code y una terminal PowerShell en la carpeta del proyecto.

Crear entorno virtual:

```powershell
py -3.12 -m venv .venv
```

Activarlo:

```powershell
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación, podés usar:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

y ejecutar directamente ese Python sin activar el entorno.

Instalar dependencias:

```powershell
python -m pip install -r requirements.txt
```

## 3. Configurar las API keys

Copiá:

```text
.env.example
```

como:

```text
.env
```

Después completá solamente la key del proveedor que quieras probar.

Ejemplo:

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=tu_clave_real
```

El `.gitignore` ya evita que `.env` se suba a Git.

## 4. Ejecutar

Con el entorno virtual activo:

```powershell
python main.py
```

El programa ejecutará dos pruebas:

1. Generación normal: espera la respuesta completa.
2. Streaming: imprime los fragmentos a medida que llegan.

## 5. Cambiar de proveedor

No hace falta cambiar el código.

### OpenAI

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-5.5
OPENAI_API_KEY=...
```

### Anthropic

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=...
```

### Groq

```env
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=...
```

### Gemini

```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=...
```

Los nombres de modelo pueden cambiar con el tiempo; si un proveedor rechaza un modelo, usá un modelo actualmente disponible en la cuenta/API correspondiente.

## 6. Estructura

```text
unified-async-llm-client/
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── schemas.py
├── base_client.py
├── clients.py
├── manager.py
└── utils.py
```

### `schemas.py`

Contiene los modelos Pydantic:

- `ChatMessage`
- `ModelConfig`
- `ModelResponse`
- `Provider`

`ModelConfig` valida, entre otras cosas, que `temperature` esté entre 0 y 2.

### `base_client.py`

Define `BaseLLMClient`, la interfaz común:

```python
async def generate(...)
async def stream(...)
```

### `clients.py`

Contiene las implementaciones concretas:

```text
OpenAIClient
AnthropicClient
GroqClient
GeminiClient
```

Todas usan clientes asíncronos.

### `manager.py`

`AsyncLLMManager` es el punto de entrada de la aplicación.

```python
manager = AsyncLLMManager(config)

response = await manager.generate(messages)

async for chunk in manager.stream(messages):
    print(chunk)
```

### `utils.py`

Normaliza errores de red, timeout, autenticación y rate limiting para que la aplicación no termine abruptamente.

## 7. Por qué es realmente asíncrono

OpenAI y Groq utilizan sus clientes async para `chat.completions.create`, Anthropic utiliza `AsyncAnthropic` y Gemini utiliza el cliente `aio`. El streaming se consume con `async for`.

Esto evita introducir llamadas síncronas al event loop principal.

## 8. Streaming

El contrato común es:

```python
async for chunk in manager.stream(messages):
    print(chunk, end="")
```

Cada proveedor transforma su propio formato de streaming en un `str` común.

Eso significa que `main.py` no necesita saber si el texto viene de OpenAI, Anthropic, Groq o Gemini.

## 9. Manejo de errores

Los errores se convierten a categorías comunes:

```text
network
rate_limit
authentication
provider_error
```

En generación normal se devuelve un `ModelResponse` con:

```python
ok=False
error_type="rate_limit"
error="..."
```

En streaming, si aparece un error después de iniciado el stream, se entrega un mensaje controlado con formato:

```text
[ERROR:rate_limit] ...
```

Así no se hace un `raise` sin controlar desde el loop principal.

## 10. Subirlo a GitHub

Inicializar Git:

```powershell
git init
git add .
git commit -m "Initial unified async LLM client"
```

Crear un repositorio vacío en GitHub, por ejemplo:

```text
unified-async-llm-client
```

Después GitHub te mostrará los comandos para conectar el repositorio local. Serán similares a:

```powershell
git branch -M main
git remote add origin https://github.com/TU_USUARIO/unified-async-llm-client.git
git push -u origin main
```

Antes de hacer `git push`, verificá:

```powershell
git status
```

`.env` **no debe aparecer** entre los archivos que Git va a subir.

## 11. Arquitectura

```text
                    main.py
                       │
                       ▼
              AsyncLLMManager
                       │
             ┌─────────┴─────────┐
             │                   │
       generate()             stream()
             │                   │
             ▼                   ▼
        BaseLLMClient
             │
     ┌───────┼────────┬──────────┐
     ▼       ▼        ▼          ▼
  OpenAI  Anthropic  Groq      Gemini
   async    async    async      async
```

El objetivo es que la aplicación dependa de la interfaz, no de un proveedor específico.

## 12. Seguridad

Nunca hagas esto:

```python
client = AsyncOpenAI(api_key="sk-...")
```

La key debe vivir en `.env` y ser leída mediante variables de entorno.

Nunca subas `.env` a GitHub.
