import numpy as np
import gradio as gr
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from typing import Tuple

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

        # Actions: 0=scale down, 1=do nothing, 2=scale up, 3=green mode
        if action == 0:
            self.state.nodes = max(1, self.state.nodes - 1)
        elif action == 2:
            self.state.nodes += 1

        self.state.load = 0.5 + 0.3 * np.sin(self.t / 5)
        self.state.latency = (self.state.load / self.state.nodes) * 150
        self.state.cost = (self.state.nodes * 1.2) * (0.7 if action == 3 else 1.0)
        self.state.carbon = (400 - (100 if action == 3 else 0))

        done = self.t >= 20
        reward = -self.state.latency  # simple reward

        return self.state, reward, done, {}

# --- API SETUP ---
app = FastAPI()   # ✅ FIXED
env = GreenOpsEnv()  # ✅ FIXED

# --- REQUEST MODEL ---
class ActionRequest(BaseModel):
    action: int

# ✅ RESET (POST)
@app.post("/reset")
def reset():
    state = env.reset()
    return {"state": state.dict()}

# ✅ STEP (POST)
@app.post("/step")
def step(req: ActionRequest):
    state, reward, done, _ = env.step(req.action)
    return {
        "state": state.dict(),
        "reward": reward,
        "done": done
    }

# ✅ STATE (GET)
@app.get("/state")
def get_state():
    return {"state": env.state.dict()}

# --- GRADIO UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🌿 OpenEnv GreenOps API Running")
    gr.Markdown("Use /reset, /step, /state endpoints")

app = gr.mount_gradio_app(app, demo, path="/")

# --- MAIN ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
