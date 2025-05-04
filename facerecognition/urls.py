"""facerecognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from django.conf.urls.static import static

from . import views

urlpatterns = [
    path('register/', views.student_registration, name='student_registration'),
    path('login/', views.student_login, name='student_login'),
    path('profile/', views.student_profile, name='student_profile'),
    path('attendance/', views.student_attendance, name='student_attendance'),
    
    # Add the path for the edit profile view
    path('profile/edit/', views.edit_profile, name='edit_profile'),
      # Custom admin panel for attendance

    path('admin/login/', views.admin_login, name='admin_login'),
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),  # Admin dashboard
    path('admin/students/', views.admin_students, name='admin_students'),
    path('admin/admin_attendance/', views.admin_attendance, name='admin_attendance'),
    path('admin/export-attendance-csv/', views.export_attendance_csv, name='export_attendance_csv'),
    path('admin/mark_attendance/', views.mark_attendance, name='mark_attendance'),
    path('mark-attendance/', views.mark_attendance_view, name='mark_attendance_view'),
    path('admin/', admin.site.urls),
      # Django Admin Panel
    path('', views.home, name='home'),  # Student module
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
     

