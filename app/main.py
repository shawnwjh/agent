from fastapi import FastAPI
app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/")
def root():
    return {"service": "agent", "status": "up"}