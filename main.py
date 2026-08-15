from fastapi import FastAPI
from router.prediction import router

app = FastAPI()

app.include_router(router)