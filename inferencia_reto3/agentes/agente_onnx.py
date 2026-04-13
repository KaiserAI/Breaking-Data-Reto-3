import onnxruntime as ort
import numpy as np
import cv2
from interfaz import AgenteBase
from collections import deque

class AgenteONNX(AgenteBase):
    def __init__(self):
        super().__init__()
        # 1. Cargamos el motor de ONNX
        self.session = ort.InferenceSession("agentes/ia_street_fighter.onnx")
        
        # 2. Inicializamos el stack de 4 frames
        self.stack = deque(maxlen=4)
        for _ in range(4):
            self.stack.append(np.zeros((84, 84), dtype=np.float32))

    def _preprocesar(self, obs, player_id):
        # Gris y redimensionar
        gris = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        res = cv2.resize(gris, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Espejado para el Jugador 2
        if player_id == 2:
            res = cv2.flip(res, 1) 
            
        return res.astype(np.float32) / 255.0

    def decidir_movimiento(self, obs, info, player_id):
        # 1. Añadir nuevo frame al stack
        self.stack.append(self._preprocesar(obs, player_id))
        
        # 2. Convertir stack a array de (1, 4, 84, 84)
        # Importante: el stack es (4, 84, 84), expandimos a (1, 4, 84, 84)
        stack_arr = np.expand_dims(np.array(self.stack), axis=0)

        # 3. Inferencia
        inputs = {self.session.get_inputs()[0].name: stack_arr}
        logits = self.session.run(None, inputs)[0][0] 
        
        # 4. Convertir Logits a botones
        # Como tu modelo tiene 12 salidas, cada índice 'i' es un botón.
        botones = [0] * 12
        for i in range(12):
            # Si el logit es positivo, pulsamos el botón
            if logits[i] > 0: 
                botones[i] = 1
        
        # --- INVERSIÓN DE CONTROLES PARA P2 ---
        # Si la IA cree que debe ir a la derecha (P1: Avanzar), 
        # al ser P2 debe pulsar izquierda para avanzar hacia el centro.
        if player_id == 2:
            izq, der = botones[6], botones[7]
            botones[6], botones[7] = der, izq
            
        return botones
