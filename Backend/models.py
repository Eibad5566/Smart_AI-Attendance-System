from sqlalchemy import Column, Integer, String, LargeBinary, Date, JSON
from database import Base

class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    student_number = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    password = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    profile_photo_path = Column(String, nullable=True) # <-- NEW: Saves the photo

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer)
    course_name = Column(String, nullable=True) # <-- NEW: Tracks the specific class
    date = Column(Date)
    status = Column(String)

class Teacher(Base):
    __tablename__ = "teachers"
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    department = Column(String, nullable=False)
    faculty_id = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False) # <-- Simplified to just 'password'
    subjects = Column(JSON, nullable=False) 
    profile_photo_path = Column(String, nullable=True)