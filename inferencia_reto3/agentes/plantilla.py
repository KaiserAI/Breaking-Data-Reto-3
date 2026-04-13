from interfaz import AgenteBase
import random

class MiAgente(AgenteBase):
    def __init__(self):
        super().__init__()
        print("Agente cargado correctamente.")
        
    def decidir_movimiento(self, obs, info, player_id):
        # Ejemplo: IA Machacabotones
        botones = [0] * 12
        if random.random() > 0.8: botones[0] = 1 # Puñetazo
        if random.random() > 0.5: botones[7] = 1 # Derecha
        return botones
