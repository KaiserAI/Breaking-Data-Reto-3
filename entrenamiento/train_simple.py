import gymnasium as gym
import ale_py  # <-- Importante: El emulador
from stable_baselines3 import PPO
import torch

def train():
    print("🚀 Iniciando entrenamiento rápido sobre RAM...")
    
    # <-- Registramos los entornos de Atari
    gym.register_envs(ale_py)

    # Creamos el entorno de Boxing versión RAM
    env = gym.make("ALE/Boxing-v5", obs_type="ram")

    # Usamos PPO (Proximal Policy Optimization) con Multi-Layer Perceptron (MlpPolicy)
    model = PPO(
        "MlpPolicy",   
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048
    )

    print("🥊 Entrenando...")
    model.learn(total_timesteps=10000)

    model.save("boxing_model_ppo")
    print("✅ Modelo guardado como 'boxing_model_ppo.zip'")

    # --- EXPORTACIÓN A ONNX ---
    print("📦 Exportando a ONNX para la carpeta de inferencia...")
    
    device = torch.device("cpu")
    model.policy.to(device)

    # Creamos una red pura de PyTorch extrayendo las piezas de SB3
    class OnnxablePolicy(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            # 1. Extractor de características (aplana la RAM)
            self.features_extractor = policy.features_extractor
            # 2. Red principal (MlpExtractor)
            self.mlp_extractor = policy.mlp_extractor
            # 3. Capa final que decide la acción
            self.action_net = policy.action_net

        def forward(self, observation):
            # Pasamos los datos capa por capa manualmente
            features = self.features_extractor(observation)
            
            # mlp_extractor devuelve latentes de política (pi) y valor (vf)
            # Solo nos interesa la política para jugar
            latent_pi, latent_vf = self.mlp_extractor(features)
            
            # Obtenemos las probabilidades (logits) de las 18 acciones
            logits = self.action_net(latent_pi)
            
            # Devolvemos el índice de la acción con mayor probabilidad (argmax)
            # Esto equivale a usar deterministic=True, pero en PyTorch puro
            return torch.argmax(logits, dim=1)

    onnxable_model = OnnxablePolicy(model.policy)
    
    # El dummy input debe coincidir con el tipo esperado (float32)
    dummy_input = torch.zeros(1, 128, dtype=torch.float32).to(device)
    
    torch.onnx.export(
        onnxable_model,
        dummy_input,
        "modelo_boxing.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['action']
    )
    print("🔥 ¡Listo! Tienes tu 'cerebro' en modelo_boxing.onnx")
    
if __name__ == "__main__":
    train()
