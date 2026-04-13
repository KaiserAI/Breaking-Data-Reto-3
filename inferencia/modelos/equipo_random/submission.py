import numpy as np
from interfaz import AgenteBase
import time

class AgenteInferencia(AgenteBase):
    """
    Este es el agente que el sistema buscará importar.
    DEBE llamarse 'AgenteInferencia' y DEBE heredar de 'AgenteBase'.
    """
    def __init__(self):
        # Llama al inicializador de la clase padre con el nombre de tu equipo
        super().__init__(nombre_equipo="Los Randoms")

    def configurar(self):
        # Como este agente tira golpes al azar, no necesitamos cargar ningún modelo ONNX.
        print(f"[{self.nombre_equipo}] Preparando guantes... ¡Listo para la acción aleatoria!")

    def predict(self, estado):
        # Ignoramos la imagen y la RAM (estado["imagen"] y estado["ram"]).
        # Simplemente devolvemos un número aleatorio entre 0 (NOOP) y 17.
        accion = np.random.randint(0, 18)
        return int(accion)
