import importlib.util
import os
import sys
import numpy as np
import time
from pettingzoo.atari import boxing_v2

# --- CONFIGURACIÓN DE REGLAS ---
LIMITE_MS = 25.0 
ACCION_PENALIZACION = 0 

def cargar_agente_desde_carpeta(ruta_carpeta):
    ruta_absoluta = os.path.abspath(ruta_carpeta)
    nombre_equipo_carpeta = os.path.basename(ruta_absoluta)
    nombre_modulo = f"modulo_{nombre_equipo_carpeta}"
    
    ruta_script = os.path.join(ruta_absoluta, "submission.py")
    if not os.path.exists(ruta_script):
        raise FileNotFoundError(f"No se encontró submission.py en {ruta_absoluta}")

    spec = importlib.util.spec_from_file_location(nombre_modulo, ruta_script)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nombre_modulo] = modulo
    
    sys.path.insert(0, ruta_absoluta)
    try:
        spec.loader.exec_module(modulo)
    finally:
        sys.path.remove(ruta_absoluta)
    
    return modulo.AgenteInferencia()

def extraer_ram_segura(env):
    unwrapped = env.unwrapped
    try:
        return unwrapped.ale.getRAM()
    except AttributeError:
        return np.zeros(128, dtype=np.uint8)

def torneo(equipo_1, equipo_2):
    agente_blanco = cargar_agente_desde_carpeta(f"modelos/{equipo_1}")
    agente_negro = cargar_agente_desde_carpeta(f"modelos/{equipo_2}")
    
    env = boxing_v2.env(render_mode="human", obs_type="rgb_image")
    env.reset()
    
    ias = {'first_0': agente_blanco, 'second_0': agente_negro}
    recompensas = {'first_0': 0, 'second_0': 0}
    stats_lentitud = {'first_0': 0, 'second_0': 0}
    total_steps = 0
    
    print(f"\n🥊 ¡COMBATE INICIADO! (Límite: {LIMITE_MS}ms) 🥊")
    

    for agent_id in env.agent_iter():
        obs, reward, term, trunc, info = env.last()
        recompensas[agent_id] += reward
        
        if term or trunc:
            action = None
        else:
            ram = extraer_ram_segura(env)
            estado_completo = {
                "imagen": obs,
                "ram": ram,
                "soy_blanco": (agent_id == 'first_0')
            }
            
            t_inicio = time.perf_counter()
            try:
                # Obtenemos la acción original de la IA
                i = time.time()
                accion_propuesta = ias[agent_id].predict(estado_completo)
                action = accion_propuesta
                f = time.time()
                print(f-i)
            except Exception as e:
                print(f"💥 Error en {ias[agent_id].nombre_equipo}: {e}")
                action = ACCION_PENALIZACION
            
            duracion_ms = (time.perf_counter() - t_inicio) * 1000
            
            # --- LOG DE PENALIZACIÓN ---
            if duracion_ms > LIMITE_MS:
                stats_lentitud[agent_id] += 1
                action = ACCION_PENALIZACION
                print(f"🛑 [PENALIZACIÓN] {ias[agent_id].nombre_equipo} fue lento: {duracion_ms:.2f}ms (Acción {accion_propuesta} descartada)")
            
            # Debug opcional: Descomenta la línea de abajo para ver el tiempo de CADA frame
            # print(f"⏱️ {ias[agent_id].nombre_equipo}: {duracion_ms:.2f}ms")

        env.step(action)
        total_steps += 1
        
    env.close()
    print(f"\n🏆 RESULTADO: {agente_blanco.nombre_equipo} {recompensas['first_0']} - {recompensas['second_0']} {agente_negro.nombre_equipo}")

if __name__ == "__main__":
    torneo("equipo_vision", "equipo_vision")
