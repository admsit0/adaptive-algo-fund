from stable_baselines3 import PPO
import torch.nn as nn

def create_agent(env, verbose=1):
    print("🧠 Inicializando Agente con arquitectura profunda...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=verbose,
        learning_rate=0.0002,    # Más lento y preciso
        n_steps=2048,            # Batch grande para estabilizar gradiente
        batch_size=64,
        ent_coef=0.01,           # Curiosidad para explorar clusters
        gamma=0.99,              # Visión a largo plazo
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]), # Red más profunda
            activation_fn=nn.Tanh # Tanh funciona mejor en regresión continua
        )
    )
    return model
