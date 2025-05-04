from django.db import models
from django.contrib.auth.models import User
import os

class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    roll_no = models.CharField(max_length=20)
    stream = models.CharField(max_length=100)
    admission_number = models.CharField(max_length=50)

    def __str__(self):
        return self.user.username

    @property
    def get_photo_directory(self):
        """Returns the directory where student photos are stored."""
        return os.path.join('dataset', self.user.username, 'photos')


class StudentPhoto(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    photo = models.ImageField(upload_to='student_photos/')
    date_captured = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Photo of {self.student.user.username} - {self.date_captured}"


class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    date = models.DateField(auto_now_add=True)
    status = models.CharField(max_length=10)

    def __str__(self):
        return f"{self.student.user.username} - {self.date} - {self.status}"
