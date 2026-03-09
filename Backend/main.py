from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime, date
import shutil
import os
import io
import pickle
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import json

from database import SessionLocal, engine
from models import Base, Student, Teacher, Attendance
from face_service import get_embeddings_and_boxes

# Create tables

Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500", "http://localhost:5500", 
        "http://127.0.0.1:8000", "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mount the uploads folder to serve profile pictures
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ===============================
# 🔐 TEACHER REGISTER & LOGIN
# ===============================
@app.post("/register_teacher")
def register_teacher(
    first_name: str = Form(...), last_name: str = Form(...), department: str = Form(...),
    faculty_id: str = Form(...), password: str = Form(...), subjects: str = Form(...),
    profile_photo: UploadFile = File(...), db: Session = Depends(get_db),
):
    if db.query(Teacher).filter(Teacher.faculty_id == faculty_id).first():
        return {"error": "A teacher with this Faculty ID is already registered."}

    try:
        subjects_list = json.loads(subjects)
    except:
        subjects_list = []

    ext = profile_photo.filename.split(".")[-1]
    file_path = f"uploads/{faculty_id}_profile.{ext}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(profile_photo.file, buffer)

    teacher = Teacher(
        first_name=first_name, last_name=last_name, department=department,
        faculty_id=faculty_id, password=password, 
        subjects=subjects_list, profile_photo_path=file_path
    )
    db.add(teacher)
    db.commit()
    return {"message": "Teacher registered successfully"}


@app.post("/login_teacher")
def login_teacher(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    teacher = db.query(Teacher).filter(Teacher.faculty_id == username).first()
    
    if not teacher or teacher.password != password:
        return {"error": "Invalid Faculty ID or Password"}
        
    return {
        "message": "Login successful",
        "teacher": {
            "first_name": teacher.first_name, "last_name": teacher.last_name,
            "department": teacher.department, "faculty_id": teacher.faculty_id,
            "subjects": teacher.subjects, "photo_url": f"http://127.0.0.1:8000/{teacher.profile_photo_path}"
        }
    }

# ===============================
# 👤 STUDENT REGISTER & LOGIN
# ===============================
@app.post("/register")
async def register_student(
    name: str = Form(...), student_number: str = Form(...), password: str = Form(...), 
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    if db.query(Student).filter(Student.student_number == student_number).first():
        return {"error": "Student number already registered."}

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    embeddings, boxes = get_embeddings_and_boxes(image)

    if len(embeddings) == 0: return {"error": "No face detected in image"}
    if len(embeddings) > 1: return {"error": "Multiple faces detected. Please upload an image with only one face."}

    ext = file.filename.split(".")[-1]
    file_path = f"uploads/student_{student_number}.{ext}"
    image.save(file_path)

    emb_bytes = pickle.dumps(embeddings[0])
    student = Student(
        name=name, student_number=student_number, password=password, 
        embedding=emb_bytes, profile_photo_path=file_path
    )
    db.add(student)
    db.commit()
    return {"message": "Student registered successfully!"}

@app.post("/login_student")
def login_student(student_number: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.student_number == student_number).first()
    
    if not student or student.password != password:
        return {"error": "Invalid Student Number or Password"}
        
    return {
        "message": "Login successful",
        "student": {
            "id": student.id, 
            "name": student.name, 
            "student_number": student.student_number,
            "photo_url": f"http://127.0.0.1:8000/{student.profile_photo_path}"
        }
    }

@app.get("/student_attendance/{student_id}")
def get_student_attendance(student_id: int, db: Session = Depends(get_db)):
    records = db.query(Attendance).filter(Attendance.student_id == student_id).all()
    # 👇 Now returning the course name as well
    return {"attendance": [{"date": r.date, "course": r.course_name, "status": r.status} for r in records]}

# ===============================
# 📸 LIVE CAMERA ATTENDANCE
# ===============================
@app.post("/attendance")
async def mark_attendance(course: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    embeddings, valid_boxes = get_embeddings_and_boxes(image)
    
    present_students = []
    face_results = [] 

    if embeddings:
        students = db.query(Student).all()
        db_embeddings = [pickle.loads(s.embedding) for s in students]

        for i, emb in enumerate(embeddings):
            best_match_idx, best_sim = -1, -1
            for j, db_emb in enumerate(db_embeddings):
                sim = cosine_similarity([emb], [db_emb])[0][0]
                if sim > best_sim:
                    best_sim = sim
                    best_match_idx = j

            xmin, ymin, xmax, ymax = valid_boxes[i]
            box_coords = {"x": int(xmin), "y": int(ymin), "width": int(xmax - xmin), "height": int(ymax - ymin)}

            if best_sim > 0.75:
                matched_student = students[best_match_idx]
                
                # --- 💾 DATABASE SAVING LOGIC STARTS HERE ---
                today = datetime.now().date()
                
                # Check if this student is already marked for this course today
                existing_record = db.query(Attendance).filter(
                    Attendance.student_id == matched_student.id,
                    Attendance.date == today,
                    Attendance.course_name == course
                ).first()

                if not existing_record:
                    new_attendance = Attendance(
                        student_id=matched_student.id,
                        course_name=course,
                        date=today,
                        status="Present"
                    )
                    db.add(new_attendance)
                    db.commit() # This actually saves it to PostgreSQL
                # --- 💾 DATABASE SAVING LOGIC ENDS HERE ---

                if matched_student.name not in present_students:
                    present_students.append(matched_student.name)

                face_results.append({
                    "box": box_coords,
                    "status": "registered",
                    "name": matched_student.name
                })
            else:
                face_results.append({
                    "box": box_coords,
                    "status": "unrecognized",
                    "name": "Unknown"
                })

    return {
        "faces": face_results,
        "present_students": present_students,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
# ===============================
# 🔐 PASSWORD RECOVERY (FORGOT PASSWORD)
# ===============================
@app.post("/reset_password")
def reset_password(
    role: str = Form(...),
    user_id: str = Form(...),
    verification_data: str = Form(...),
    new_password: str = Form(...),
    db: Session = Depends(get_db)
):
    if role == "student":
        student = db.query(Student).filter(Student.student_number == user_id).first()
        # Verify student exists and the name exactly matches (case-insensitive)
        if not student or student.name.strip().lower() != verification_data.strip().lower():
            return {"error": "Verification failed. Invalid Student ID or Full Name."}
        
        student.password = new_password
        db.commit()
        return {"message": "Student password reset successfully!"}
        
    elif role == "teacher":
        teacher = db.query(Teacher).filter(Teacher.faculty_id == user_id).first()
        # Verify teacher exists and the department exactly matches
        if not teacher or teacher.department.strip().lower() != verification_data.strip().lower():
            return {"error": "Verification failed. Invalid Faculty ID or Department."}
        
        teacher.password = new_password
        db.commit()
        return {"message": "Teacher password reset successfully!"}
        
    return {"error": "Invalid role specified."}