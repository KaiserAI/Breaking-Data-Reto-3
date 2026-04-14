# Hackathon: Atari Boxing AI

Este repositorio proporciona el entorno de desarrollo y la arena de competición oficial para entrenar y evaluar agentes en el juego **Boxing** de Atari 2600. El objetivo es programar un agente que se enfrentará a otros en un torneo de eliminación.

## Estructura del Proyecto

El sistema está dividido en dos entornos aislados:
1. **`/entrenamiento`**: Tu laboratorio. Aquí puedes instalar lo que quieras, usar PyTorch, TensorFlow, o scripts de recolección de datos. Contiene ejemplos básicos funcionales (`train_simple.py` y `train_vision.py`).
2. **`/inferencia`**: La Arena oficial. Es un entorno bloqueado y estandarizado donde los agentes compiten.

---

## 1. Configuración del Entorno (`uv`)

Utilizamos `uv` como gestor de proyectos para garantizar que todos los participantes y el motor del torneo corran exactamente las mismas versiones y eviten errores de compilación.

### Preparar la Arena (Inferencia)
Entra en la carpeta `inferencia` y ejecuta:
```bash
uv sync
uv run AutoROM --accept-license
```
*¿Qué hace esto?* Lee el archivo `pyproject.toml`, descarga una versión aislada de Python 3.9 y monta el entorno virtual `.venv` con las librerías permitidas.

⚠️ **REGLA ESTRICTA DE INFERENCIA**: En la carpeta `/inferencia` **NO SE PUEDE INSTALAR NADA NUEVO**. Las librerías disponibles están fijadas en el `pyproject.toml` (numpy, onnxruntime, opencv-headless, litellm, etc.). Si tu script intenta importar una librería no incluida aquí, tu agente fallará al cargar y será descalificado.

---

## 2. Tipos de Agentes Permitidos

Puedes construir el "cerebro" de tu boxeador utilizando diferentes enfoques. El entorno soporta las siguientes arquitecturas:

| Tipo de IA | Archivo de "Cerebro" | Librería permitida en Inferencia |
| :--- | :--- | :--- |
| **Deep Learning (RL / CNN)** | `.onnx` | `onnxruntime` |
| **LLM (Agentes / Prompts)** | `.env` (con tu API Key) | `litellm` o `requests` |
| **Reglas / Heurística** | Ninguno (o un `.json` / `.py` extra) | Python puro / `numpy` |

---

## 3. Construyendo tu Agente

Para participar, crea una carpeta con el nombre de tu equipo dentro de `inferencia/modelos/` (ej. `inferencia/modelos/equipo_alfa/`).

Dentro, debes crear un archivo llamado **`submission.py`**. Este archivo debe contener una clase `AgenteInferencia` que herede de `AgenteBase` (ubicada en `interfaz.py`).

### Estructura base (`submission.py`)
```python
import os
import numpy as np
from interfaz import AgenteBase

class AgenteInferencia(AgenteBase):
    def __init__(self):
        super().__init__(nombre_equipo="Nombre de tu Equipo")

    def configurar(self):
        # Se ejecuta una sola vez al cargar. 
        # Úsalo para cargar modelos .onnx o leer tu archivo .env
        pass

    def predict(self, estado):
        # Lógica principal. Debe devolver un entero del 0 al 5.
        return 1
```

### Ejemplos de implementación según tu enfoque:

**A. Enfoque por Reglas (Heurística):**
No requiere modelo externo. Extraes las coordenadas de la RAM e implementas lógica condicional para acercarte al rival y golpear. Muy rápido (0.1ms), ideal para asegurar que nunca serás penalizado por tiempo.

**B. Enfoque con LLMs (LiteLLM):**
Si decides pasarle el estado del juego a un modelo de lenguaje, debes incluir un archivo `.env` en tu carpeta con tus credenciales. En tu método `configurar()`, utiliza la librería `python-dotenv` para cargarlas de forma segura. 
*Nota crítica:* Depender de APIs externas (OpenAI, Anthropic) tiene una alta latencia. Revisa la sección de Penalizaciones.

**C. Enfoque Deep Learning (ONNX):**
Exporta tu modelo entrenado a `.onnx`. Utiliza `onnxruntime.InferenceSession` en el método `configurar()` para cargarlo en memoria. La predicción procesará la matriz de píxeles o la RAM.

---

## 4. Entrada y Salida (El diccionario `estado`)

En cada frame, el método `predict(self, estado)` recibe un diccionario con la siguiente estructura:

* **`estado["ram"]`**: Array Numpy de 128 bytes (uint8) con la memoria de la consola.
* **`estado["imagen"]`**: Array Numpy tridimensional con la imagen RGB de la pantalla (210x160 píxeles).
* **`estado["soy_blanco"]`**: Booleano. `True` si controlas al boxeador superior (blanco), `False` si controlas al inferior (negro). **Vital para que tu agente sepa en qué dirección moverse.**

### Mapeo de la Memoria RAM
Si usas heurística o redes neuronales basadas en RAM, estas direcciones son interesantes:

| Dirección | Valor |
| :--- | :--- |
| `ram[32]` | Posición X del boxeador Blanco |
| `ram[34]` | Posición Y del boxeador Blanco |
| `ram[33]` | Posición X del boxeador Negro |
| `ram[35]` | Posición Y del boxeador Negro |
| `ram[18]` | Puntuación del jugador Blanco |
| `ram[19]` | Puntuación del jugador Negro |
| `ram[17]` | Tiempo restante del round (reloj) |

### Salida esperada (Acciones)
Tu método debe devolver un número entero (0-5):
* `0`: NOOP (Quieto)
* `1`: GOLPEAR
* `2`: MOVER ARRIBA
* `3`: MOVER DERECHA
* `4`: MOVER IZQUIERDA
* `5`: MOVER ABAJO

---

## 5. El Motor del Torneo y Penalizaciones

Para probar tus modelos, modifica el final del archivo `arena.py` con los nombres de tus carpetas y ejecuta:
```bash
uv run arena.py
```

### Límite de Inferencia Estricto (25ms)
La inferencia tiene un límite de **25.0 milisegundos por frame**. El árbitro (`arena.py`) cronometra internamente la llamada a tu método `predict`.

* Si tu agente responde en menos de 25ms: La acción se ejecuta normalmente.
* Si tu agente tarda **más de 25ms**: Verás un log en la terminal indicando `🛑 [PENALIZACIÓN]`. La acción que hayas devuelto se descarta y el árbitro fuerza la acción `0` (quedarse quieto).

*Aviso para usuarios de LLMs:* Una llamada típica a una API externa tarda entre 300ms y 1000ms. Si usas LLMs de forma síncrona en cada frame, tu boxeador pasará el 100% del tiempo penalizado. Necesitarás pensar en procesamiento local o estrategias asíncronas permitidas.
