import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import torch

def train_vision():
    print("🎥 Iniciando entrenamiento con Visión Artificial (CNN)...")
    gym.register_envs(ale_py)

    # 1. Creamos el entorno con los wrappers automáticos de Atari
    # Esto hace el resize a 84x84 y escala de grises automáticamente
    env = make_atari_env("ALE/Boxing-v5", n_envs=1, seed=0)
    
    # 2. Frame Stacking: Le damos 4 frames de memoria para que vea el movimiento
    env = VecFrameStack(env, n_stack=4)

    # 3. Usamos CnnPolicy en lugar de MlpPolicy
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0001, # Un poco más bajo para CNN
        n_steps=128,          # Pasos más cortos para actualizar más rápido
        batch_size=256,
        n_epochs=4
    )

    print("🥊 Entrenando el ojo del tigre...")
    # Para visión, 10,000 pasos es muy poco, pero sirve para probar el código.
    # Un agente decente necesita 1,000,000.
    model.learn(total_timesteps=20000)

    model.save("boxing_vision_model")
    
    # --- EXPORTACIÓN A ONNX ---
    print("📦 Exportando CNN a ONNX...")
    device = torch.device("cpu")
    model.policy.to(device)

    class OnnxableVisionPolicy(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.features_extractor = policy.features_extractor
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, observation):
            # La observación viene como (Batch, 4, 84, 84)
            features = self.features_extractor(observation)
            latent_pi, _ = self.mlp_extractor(features)
            logits = self.action_net(latent_pi)
            return torch.argmax(logits, dim=1)

    onnxable_model = OnnxableVisionPolicy(model.policy)
    # Dummy input: 1 imagen, 4 frames de stack, 84x84 pixeles
    dummy_input = torch.randn(1, 4, 84, 84).to(device)
    
    torch.onnx.export(
        onnxable_model,
        dummy_input,
        "modelo_vision.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['action']
    )
    print("🔥 ¡Listo! Tienes tu cerebro visual en modelo_vision.onnx")

if __name__ == "__main__":
    train_vision()
