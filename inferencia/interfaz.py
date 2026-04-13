import os
import sys

class AgenteBase:
    def __init__(self, nombre_equipo):
        self.nombre_equipo = nombre_equipo
        
        # BUSQUEDA ROBUSTA: Accedemos al atributo __file__ del módulo cargado
        # self.__module__ contiene el nombre que le dimos en arena.py (ej: 'modulo_equipo_alfa')
        nombre_modulo = self.__class__.__module__
        
        if nombre_modulo in sys.modules:
            archivo_hijo = sys.modules[nombre_modulo].__file__
            self.ruta_equipo = os.path.dirname(os.path.abspath(archivo_hijo))
        else:
            # Fallback por si se ejecuta de forma local o directa
            self.ruta_equipo = os.getcwd()
        
        self.configurar()

    def configurar(self):
        """Método para que el equipo cargue sus modelos (ONNX, etc.)"""
        pass

    def predict(self, estado):
        """Método que el equipo debe implementar"""
        raise NotImplementedError("Falta implementar predict()")
