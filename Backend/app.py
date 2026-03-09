import io
import sqlite3
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Face detection
mtcnn = MTCNN(keep_all=False, device=device)

# Face recognition
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Database
conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    student_number TEXT UNIQUE,
    class_name TEXT,
    email TEXT,
    password TEXT,
    embedding BLOB
)
""")
conn.commit()

def extract_embedding(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    face = mtcnn(image)

    if face is None:
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = resnet(face)

    return embedding.cpu().numpy()[0]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.get("/")
def root():
    return {"status": "Real Smart AI Attendance Running"}

@app.post("/register_student")
async def register_student(
    name: str = Form(...),
    student_number: str = Form(...),
    class_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    image: UploadFile = File(...)
):
    image_bytes = await image.read()
    embedding = extract_embedding(image_bytes)

    if embedding is None:
        return JSONResponse(status_code=400, content={"error": "No face detected"})

    try:
        cursor.execute(
            "INSERT INTO students VALUES (NULL, ?, ?, ?, ?, ?, ?)",
            (name, student_number, class_name, email, password, embedding.tobytes())
        )
        conn.commit()
    except:
        return JSONResponse(status_code=400, content={"error": "Student already exists"})

    return {"status": "Student registered successfully"}

@app.post("/login_with_face")
async def login_with_face(image: UploadFile = File(...)):
    image_bytes = await image.read()
    embedding = extract_embedding(image_bytes)

    if embedding is None:
        return {"error": "No face detected"}

    cursor.execute("SELECT name, student_number, class_name, email, embedding FROM students")
    students = cursor.fetchall()

    for st in students:
        db_embedding = np.frombuffer(st[4], dtype=np.float32)
        similarity = cosine_similarity(embedding, db_embedding)

        if similarity > 0.75:
            return {
                "status": "Login success",
                "student": {
                    "name": st[0],
                    "student_number": st[1],
                    "class_name": st[2],
                    "email": st[3]
                }
            }

    return {"status": "No matching face found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)