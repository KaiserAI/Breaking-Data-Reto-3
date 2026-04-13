import stable_retro as retro
import torch
import gymnasium as gym
import numpy as np
import cv2
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

# --- 1. EL WRAPPER DE VISIÓN ---
class HackathonVisionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )
        self.health = 176
        self.enemy_health = 176

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.health = 176
        self.enemy_health = 176
        return self._procesar_imagen(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        imagen_ia = self._procesar_imagen(obs)
        
        # --- 2. LA FUNCIÓN DE RECOMPENSA ---
        # Extraemos vidas de la RAM (info viene del data.json original)
        current_health = info.get('health', self.health)
        current_enemy_health = info.get('enemy_health', self.enemy_health)
        
        # Calculamos daño infligido y recibido
        damage_inflicted = self.enemy_health - current_enemy_health
        damage_received = self.health - current_health
        
        # Recompensa: +2 por pegar, -1 por ser pegado
        # (Aconsejamos que experimenten con esto)
        custom_reward = (damage_inflicted * 4.0) - (damage_received * 1.0)
        
        if damage_inflicted == 0:
            custom_reward -= 3 # Penalización por pasividad

        # Actualizamos estado para el próximo frame
        self.health = current_health
        self.enemy_health = current_enemy_health
        
        return imagen_ia, custom_reward, terminated, truncated, info

    def _procesar_imagen(self, obs):
        gris = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        redimensionada = cv2.resize(gris, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(redimensionada, axis=-1)

# Callback sencillo para guardar el modelo cada X pasos
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
        return True

# --- 3. CONFIGURACIÓN DEL ENTRENAMIENTO ---
def main():
    # Creamos directorios para guardar datos
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'
    
    # Iniciamos el entorno (render_mode="rgb_array" para obtener píxeles)
    print("Arrancando el entorno de Street Fighter II...")
    env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis-v0', state='Champion.Level1.RyuVsGuile', # <--- MISMO QUE EN TORNEO,
                     render_mode="rgb_array") # <--- Aquí no pongan lo de human, human también devuelve rgb_array, pero también levanta la interfaz gráfica)
    
    # Aplicamos nuestro Wrapper de visión y recompensas
    env = HackathonVisionWrapper(env)
    
    # ESTO ES CRUCIAL PARA VISIÓN: FrameStack
    # La IA no ve 1 imagen, ve las últimas 4 imágenes apiladas.
    # Así puede entender el MOVIMIENTO (si un puño está subiendo o bajando).
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    
    # Configuramos el algoritmo PPO con una Red Neuronal Convolucional (CnnPolicy)
    # verbose=1 para ver el progreso en la terminal
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0003, n_steps=2048)
    
    # Configuramos el autoguardado cada 10,000 pasos
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    
    # ¡A ENTRENAR!
    # Para una PoC rápida, 50,000 pasos son suficientes para ver conducta.
    # En el hackathon, los equipos deberían apuntar a 500,000 o 1,000,000.
    print("\n--- INICIANDO ENTRENAMIENTO EXPRES (50,000 pasos) ---")
    print("Puedes pararlo en cualquier momento con Ctrl+C.")
    try:
        model.learn(total_timesteps=50000, callback=callback) #50000
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
    finally:
        # 1. Guardamos el modelo original de SB3
        model.save("ia_street_fighter_poc")
        print("Modelo SB3 guardado.")

        # 2. DEFINIR EL WRAPPER PARA ONNX
        class OnnxablePolicy(torch.nn.Module):
            def __init__(self, extractor, action_net):
                super().__init__()
                self.extractor = extractor
                self.action_net = action_net

            def forward(self, observation):
                features = self.extractor(observation)
                return self.action_net(features)

        # 3. PREPARAR EL MODELO
        print("Preparando exportación a ONNX...")
        # Extraemos las piezas y las ponemos en modo EVALUACIÓN
        # Esto quita el warning de 'training mode'
        onnx_policy = OnnxablePolicy(
            model.policy.features_extractor, 
            model.policy.action_net
        ).to("cpu")
        onnx_policy.eval() 

        dummy_input = torch.randn(1, 4, 84, 84)

        print("Exportando a ONNX...")
        try:
            torch.onnx.export(
                onnx_policy,
                dummy_input,
                "ia_street_fighter.onnx",
                opset_version=18, # Subimos a 18 para evitar errores de conversión
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print("✅ ¡Archivo 'ia_street_fighter.onnx' generado con éxito!")
        except Exception as e:
            print(f"❌ Error crítico: {e}")
        
        env.close()

if __name__ == "__main__":
    main()
