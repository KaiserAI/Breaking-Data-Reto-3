from interfaz import AgenteBase
import random

class AgenteAleatorio(AgenteBase):
    def __init__(self):
        super().__init__()
        print("🤖 Agente Aleatorio cargado")
        
    def decidir_movimiento(self, obs, info, player_id):
        botones = [0] * 12
        
        # Moverse aleatoriamente (índices 4, 5, 6, 7)
        direccion = random.choice([4, 5, 6, 7])
        botones[direccion] = 1
        
        # Atacar aleatoriamente el 60% de las veces
        if random.random() > 0.4:
            ataque = random.choice([0, 1, 2, 8, 9, 10, 11])
            botones[ataque] = 1
            
        return botones
