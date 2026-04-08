import numpy as np
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

class State(BaseModel):
    load: float
    latency: float
    cost: float
    carbon: float
    nodes: int

class ActionRequest(BaseModel):
    action: int

class GreenOpsEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.t = 0
        self.state = State(load=0.5, latency=20.0, cost=5.0, carbon=400.0, nodes=5)
        return self.state

    def step(self, action: int):
        self.t += 1

        if action == 0:
            self.state.nodes = max(1, self.state.nodes - 1)
        elif action == 2:
            self.state.nodes += 1

        self.state.load = 0.5 + 0.3 * np.sin(self.t / 5)
        self.state.latency = (self.state.load / self.state.nodes) * 150
        self.state.cost = (self.state.nodes * 1.2) * (0.7 if action == 3 else 1.0)
        self.state.carbon = (400 - (100 if action == 3 else 0))

        done = self.t >= 20
        reward = -self.state.latency

        return self.state, reward, done, {}

app = FastAPI()
env = GreenOpsEnv()

@app.post("/reset")
def reset():
    return {"state": env.reset().dict()}

@app.post("/step")
def step(req: ActionRequest):
    state, reward, done, _ = env.step(req.action)
    return {"state": state.dict(), "reward": reward, "done": done}

@app.get("/state")
def state():
    return {"state": env.state.dict()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
