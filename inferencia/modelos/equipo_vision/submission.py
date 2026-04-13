import os
import cv2
import numpy as np
import onnxruntime as ort
import collections
from interfaz import AgenteBase

class AgenteInferencia(AgenteBase):
    def __init__(self):
        super().__init__(nombre_equipo="Vision Knights")

    def configurar(self):
        # 1. Cargar ONNX
        ruta_modelo = os.path.join(self.ruta_equipo, "modelo_vision.onnx")
        
        # Optimizamos para tu Ryzen 9: limitamos hilos para evitar overhead
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        
        self.session = ort.InferenceSession(ruta_modelo, sess_options=options)
        self.input_name = self.session.get_inputs()[0].name
        
        # 2. Cola para el Frame Stacking (4 frames de memoria)
        self.stack_size = 4
        self.frame_stack = collections.deque(maxlen=self.stack_size)
        
        print(f"[{self.nombre_equipo}] 👁️ Ojo digital cargado. Listo para procesar píxeles.")

    def preprocesar(self, rgb_image):
        # Convertir a escala de grises
        gris = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        # Redimensionar a 84x84 (estándar de Atari RL)
        peque = cv2.resize(gris, (84, 84), interpolation=cv2.INTER_AREA)
        return peque

    def predict(self, estado):
        """
        estado['imagen']: Imagen RGB cruda (210x160)
        estado['soy_blanco']: True/False 
        """
        # 1. Procesar el frame actual
        frame_procesado = self.preprocesar(estado["imagen"])
        
        # --- TRUCO OPCIONAL ---
        # Si el modelo fue entrenado solo como Blanco, y ahora somos Negro,
        # podríamos invertir los colores si el equipo lo considera necesario.
        # if not estado["soy_blanco"]:
        #     frame_procesado = 255 - frame_procesado
        
        # 2. Gestionar el stack (memoria temporal para detectar movimiento)
        if not self.frame_stack:
            for _ in range(self.stack_size):
                self.frame_stack.append(frame_procesado)
        else:
            self.frame_stack.append(frame_procesado)

        # 3. Formatear para la red neuronal: (Batch=1, Canales=4, H=84, W=84)
        input_tensor = np.array(self.frame_stack, dtype=np.float32)
        input_tensor = np.expand_dims(input_tensor, axis=0) # Añadir Batch
        input_tensor /= 255.0 # Normalizar a [0, 1]

        # 4. Inferencia ONNX
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # El modelo ya devuelve el índice de la acción (argmax)
        return int(outputs[0][0])
