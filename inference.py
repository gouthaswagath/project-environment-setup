import numpy as np
import yaml
import gradio as gr
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import List, Tuple

# --- MODELS ---
class State(BaseModel):
    load: float
    latency: float
    cost: float
    carbon: float
    nodes: int

# --- ENVIRONMENT ---
class GreenOpsEnv:
    def __init__(self):
        self.reset()
    def reset(self) -> State:
        self.t = 0
        self.state = State(load=0.5, latency=20.0, cost=5.0, carbon=400.0, nodes=5)
        return self.state
    def step(self, action: int) -> Tuple[State, float, bool, dict]:
        self.t += 1
        if action == 0: self.state.nodes = max(1, self.state.nodes - 1)
        if action == 2: self.state.nodes += 1
        self.state.load = 0.5 + 0.3 * np.sin(self.t / 5)
        self.state.latency = (self.state.load / self.state.nodes) * 150
        self.state.cost = (self.state.nodes * 1.2) * (0.7 if action == 3 else 1.0)
        self.state.carbon = (400 - (100 if action == 3 else 0))
        return self.state, 0.0, self.t >= 20, {}

# --- API SETUP ---
app = FastAPI()
env = GreenOpsEnv()

# CRITICAL: This fixes the "Method Not Allowed" error
@app.post("/reset")
def reset_endpoint():
    return env.reset().dict()

@app.post("/step")
def step_endpoint(action: int):
    state, reward, done, info = env.step(action)
    return {"state": state.dict(), "reward": reward, "done": done}

# --- GRADIO UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🌿 OpenEnv GreenOps API")
    gr.Interface(fn=lambda x: "API is Running", inputs="text", outputs="text")

# Mount Gradio onto FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    # Ensure this runs on port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
