from django.contrib import admin
from django.urls import path
from django.http import HttpResponseRedirect
import cv2
import numpy as np
import os
from django.conf import settings
from .models import Student, Attendance

@admin.register(Student)
class StudentAdmin(admin.ModelAdmin):
    list_display = ('user', 'roll_no', 'stream', 'admission_number')  
    search_fields = ('user__username', 'roll_no', 'admission_number', 'stream')

    # Add custom URL for marking attendance
    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('mark_attendance/', self.admin_site.admin_view(self.mark_attendance_view), name='mark_attendance'),
        ]
        return custom_urls + urls

    def mark_attendance_view(self, request):
        # Load the face recognizer model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(os.path.join(settings.BASE_DIR, 'admin_panel/utils/face_recognizer.yml'))

        # Load the label dictionary
        label_dict = np.load(os.path.join(settings.BASE_DIR, 'admin_panel/utils/label_dict.npy'), allow_pickle=True).item()

        # Initialize the webcam for capturing photos
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        student_name = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                label, confidence = recognizer.predict(face)

                if confidence < 50:  # Confidence threshold
                    student_name = label_dict.get(label, None)
                    if student_name:
                        # Draw the rectangle and student name
                        cv2.putText(frame, student_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        break

            # Display the webcam feed
            cv2.imshow('Mark Attendance', frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or student_name:
                break  # Stop when a student is recognized or 'q' is pressed

        cap.release()
        cv2.destroyAllWindows()

        # Mark attendance if student is recognized
        if student_name:
            try:
                student = Student.objects.get(user__username=student_name)
                Attendance.objects.create(student=student, status='Present')
                return HttpResponseRedirect('/admin/student/student/')
            except Student.DoesNotExist:
                return HttpResponseRedirect('/admin/student/student/')
        else:
            return HttpResponseRedirect('/admin/student/student/')

# Register Attendance Model in Admin
@admin.register(Attendance)
class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('student', 'student_roll_no', 'student_stream', 'student_admission_number', 'date', 'status')
    search_fields = ('student__user__username', 'student__roll_no', 'student__admission_number', 'date')
    list_filter = ('status', 'student__stream', 'date')

    # Display roll number, stream, and admission number in the attendance list
    def student_roll_no(self, obj):
        return obj.student.roll_no

    def student_stream(self, obj):
        return obj.student.stream

    def student_admission_number(self, obj):
        return obj.student.admission_number

