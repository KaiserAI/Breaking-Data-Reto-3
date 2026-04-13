import numpy as np

class AgenteBase:
    def __init__(self):
        """Inicializa tu modelo o variables aquí."""
        pass
        
    def decidir_movimiento(self, obs: np.ndarray, info: dict, player_id: int) -> list[int]:
        """
        obs: Imagen RGB (224, 320, 3)
        info: Diccionario con la RAM
        player_id: Será 1 si juegas a la izquierda (P1), o 2 si juegas a la derecha (P2).
        Retorna: Lista de 12 enteros (0 o 1)
        """
        raise NotImplementedError("Debes implementar este método.")
