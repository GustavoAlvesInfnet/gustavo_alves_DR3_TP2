
from fastapi import FastAPI
import uvicorn
from routes import router as api_router

app = FastAPI()
app.include_router(api_router)

def __main__():
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
    #uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    __main__()


