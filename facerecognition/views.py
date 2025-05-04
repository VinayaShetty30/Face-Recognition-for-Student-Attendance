import json
import os
import base64
import cv2
import re
import numpy as np
import csv
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login
from django.http import JsonResponse, HttpResponse, HttpResponseRedirect
from django.conf import settings
from django.core.mail import send_mail
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.utils.timezone import now
from .models import Student, Attendance, StudentPhoto
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import base64
import numpy as np
import cv2
import os
from django.conf import settings
from .models import Student, Attendance


from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np
import cv2
import base64
import re
import os
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect
from .models import User, Student
import pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import cv2
import torch
import base64
from django.http import JsonResponse
from .models import Student, Attendance
import json
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import dlib
from train_model import train_model

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)  # Detect faces
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # Generate embeddings

EMBEDDINGS_FILE = os.path.join(settings.BASE_DIR, 'utils', 'embeddings.pickle')

# Student registration view
def student_registration(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        roll_no = request.POST.get('roll_no')
        stream = request.POST.get('stream')
        admission_number = request.POST.get('admission_number')
        captured_images = request.POST.getlist('captured_images[]')  # Fetch images properly

        sanitized_username = re.sub(r'\W+', '_', username)
        student_dir = os.path.join(settings.BASE_DIR, 'dataset', sanitized_username, 'photos')
        os.makedirs(student_dir, exist_ok=True)

        try:
            user = User.objects.create_user(username=username, password=password)
            student = Student.objects.create(user=user, roll_no=roll_no, stream=stream, admission_number=admission_number)
            student.save()

            image_paths = []
            for idx, captured_image in enumerate(captured_images, start=1):
                if captured_image:
                    format, imgstr = captured_image.split(';base64,')
                    img_data = base64.b64decode(imgstr)  # Decode Base64 image

                    file_path = os.path.join(student_dir, f'{sanitized_username}_{idx}.jpg')
                    with open(file_path, 'wb') as f:
                        f.write(img_data)

                    StudentPhoto.objects.create(student=student, photo=file_path)  # Save to DB
                    image_paths.append(file_path)
            train_model()
            
            return JsonResponse({"message": "Student registered successfully", "saved_images": image_paths})
        except Exception as e:
            return JsonResponse({"error": f"Failed to save student data: {str(e)}"})
    
    return render(request, 'student_registration.html')

# Home page
def home(request):
    return render(request, 'home.html')


# Student login
def student_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('student_profile')
        else:
            return render(request, 'student_login.html', {'error': 'Invalid username or password'})
    return render(request, 'student_login.html')


# Student profile
@login_required
def student_profile(request):
    student = get_object_or_404(Student, user=request.user)
    return render(request, 'student_profile.html', {'student': student})


# Edit student profile
@login_required
def edit_profile(request):
    student = get_object_or_404(Student, user=request.user)

    if request.method == 'POST':
        student.roll_no = request.POST.get('roll_no', student.roll_no)
        student.stream = request.POST.get('stream', student.stream)
        student.admission_number = request.POST.get('admission_number', student.admission_number)
        student.save()
        return redirect('student_profile')

    return render(request, 'edit_profile.html', {'student': student})


# Student attendance view
@login_required
def student_attendance(request):
    attendance_records = Attendance.objects.filter(student__user=request.user)
    return render(request, 'student_attendance.html', {'attendance_records': attendance_records})


# Admin login
def admin_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None and user.is_staff:
            login(request, user)
            return redirect('admin_dashboard')
    return render(request, 'admin_login.html')


# Admin dashboard
@login_required(login_url='admin_login/')
def admin_dashboard(request):
    return render(request, 'admin_dashboard.html')


# View students
@login_required(login_url='admin_login/')
def admin_students(request):
    students = Student.objects.all()
    return render(request, 'admin_students.html', {'students': students})


# View attendance records
@login_required(login_url='admin_login/')
def admin_attendance(request):
    attendance_records = Attendance.objects.all()
    return render(request, 'admin_attendance.html', {'attendance_records': attendance_records})


# CSV Export for attendance
@login_required(login_url='admin_login/')
def export_attendance_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="attendance.csv"'
    writer = csv.writer(response)
    writer.writerow(['Student', 'Date', 'Status'])
    attendance_records = Attendance.objects.all()
    for record in attendance_records:
        writer.writerow([record.student.user.username, record.date, record.status])
    return response

@login_required(login_url='admin_login')
def mark_attendance(request):
    return render(request, 'mark_attendance.html')



# Load the face alignment model


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load stored embeddings
with open(EMBEDDINGS_FILE, "rb") as f:
    embeddings_data = pickle.load(f)
    stored_embeddings = np.array(embeddings_data["embeddings"])
    stored_labels = embeddings_data["labels"]

def preprocess_image(img):
    """ Convert to grayscale and adjust brightness/contrast. """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Normalize brightness
    return gray

def get_face_embedding(img):
    """
    Extracts a 128D face embedding from an image using Dlib's face recognition model.
    Handles multiple faces and picks the **largest** detected face.
    """
    gray = preprocess_image(img)
    faces = detector(gray)

    if len(faces) == 0:
        print("⚠ No face detected - Try improving lighting or contrast")
        return None

    # Pick the largest face (useful when multiple are detected)
    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())

    shape = sp(gray, largest_face)
    return np.array(facerec.compute_face_descriptor(img, shape))

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@login_required(login_url="admin_login")
def mark_attendance_view(request):
    """
    Django view to mark attendance based on facial recognition.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            captured_image = data.get("captured_image")

            if not captured_image:
                return JsonResponse({"message": "No image received"})

            # Decode base64 image
            format, imgstr = captured_image.split(";base64,")
            img_data = base64.b64decode(imgstr)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Resize image for better detection (optional)
            img = cv2.resize(img, (160, 160))

            # Extract face embedding
            face_embedding = get_face_embedding(img)

            if face_embedding is None:
                return JsonResponse({"message": "No face detected. Please ensure good lighting & proper framing."})

            # Compare with stored embeddings
            similarities = [cosine_similarity(face_embedding, emb) for emb in stored_embeddings]
            max_similarity_index = np.argmax(similarities)
            max_similarity = similarities[max_similarity_index]

            # Debugging logs
            print(f"📌 Similarity scores: {similarities}")
            print(f"🔍 Max similarity: {max_similarity}")

            similarity_threshold = 0.50  # Adjust based on testing

            if max_similarity > similarity_threshold:
                student_name = stored_labels[max_similarity_index]
                student = Student.objects.get(user__username=student_name)

                # Mark attendance
                Attendance.objects.create(student=student, status="Present")
                return JsonResponse({"message": f"✅ Attendance marked for {student_name}"})
            else:
                return JsonResponse({"message": "No matching student found. Try again."})

        except Exception as e:
            return JsonResponse({"message": f"⚠ Error processing image: {str(e)}"})

    return JsonResponse({"message": "Invalid request method!"})
