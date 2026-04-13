import stable_retro as retro
import time
import numpy as np

from agentes.agente_onnx import  AgenteONNX as AgenteP1 
from agentes.agente_aleatorio import AgenteAleatorio as AgenteP2 

def main():
    print("🔥 INICIANDO COMBATE 🔥")
    
    p1 = AgenteP1()
    p2 = AgenteP2()
    
    # IMPORTANTE: Forzamos el state para que vayan directos al combate
    env = retro.make(
        game='StreetFighterIISpecialChampionEdition-Genesis-v0', 
        state='Champion.Level1.RyuVsGuile', 
        players=2, 
        render_mode="human"
    )
    
    obs, info = env.reset()
    
    try:
        while True:
            # Le decimos a cada IA quién es (1 o 2)
            try:
                accion_p1 = p1.decidir_movimiento(obs, info, player_id=1)
            except Exception as e:
                print(f"Error P1: {e}")
                accion_p1 = [0] * 12
                
            try:
                accion_p2 = p2.decidir_movimiento(obs, info, player_id=2)
            except Exception as e:
                print(f"Error P2: {e}")
                accion_p2 = [0] * 12

            accion_combinada = np.concatenate([accion_p1, accion_p2])
            obs, reward, terminated, truncated, info = env.step(accion_combinada)
            
            if terminated or truncated:
                print("¡K.O.!")
                obs, info = env.reset()
                time.sleep(2)
                
            time.sleep(0.016)

    except KeyboardInterrupt:
        print("\nTorneo detenido.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
