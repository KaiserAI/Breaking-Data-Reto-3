import os
import numpy as np
import onnxruntime as ort
from interfaz import AgenteBase

class AgenteInferencia(AgenteBase):
    def __init__(self):
        # Nombre identificativo para el torneo
        super().__init__(nombre_equipo="Los Tensores ONNX")

    def configurar(self):
        """
        Cargamos el modelo ONNX. Este método se ejecuta una sola vez al inicio.
        """
        ruta_modelo = os.path.join(self.ruta_equipo, "modelo_boxing.onnx")
        
        if not os.path.exists(ruta_modelo):
            raise FileNotFoundError(f"❌ No se encontró el modelo en: {ruta_modelo}")

        # Configuración de hilos para no colapsar el Ryzen 9 si hay muchos combates
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2 
        
        self.session = ort.InferenceSession(ruta_modelo, sess_options=options)
        self.input_name = self.session.get_inputs()[0].name
        
        print(f"[{self.nombre_equipo}] 🧠 Cerebro ONNX listo y conectado.")

    def predict(self, estado):
        """
        estado['ram']: 128 bytes de la consola
        estado['soy_blanco']: True si manejamos al de arriba, False para el de abajo
        """
        ram = estado["ram"]
        soy_blanco = estado["soy_blanco"]
        
        # --- LÓGICA DE APOYO ---
        # Si el equipo descubre que su modelo solo sabe pelear como 'blanco',
        # aquí podría aplicar "espejo" a la RAM antes de pasarla al modelo.
        # Por ahora, le pasamos la RAM cruda (128 bytes).
        
        # Preprocesamiento para ONNX: (Batch=1, Features=128)
        input_data = ram.astype(np.float32).reshape(1, 128)
        
        # Inferencia
        outputs = self.session.run(None, {self.input_name: input_data})
        
        # Extraemos la acción (entero)
        accion = int(outputs[0][0])
        
        # Opcional: Loguear ocasionalmente si somos el blanco o negro para debug
        # if ram[11] % 10 == 0: # Cada 10 segundos de juego
        #    color = "BLANCO" if soy_blanco else "NEGRO"
        #    print(f"[{self.nombre_equipo}] Controlando al boxeador {color}")

        return accion
