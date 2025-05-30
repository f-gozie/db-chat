"""
Django settings for db_chat_project project.

Generated by 'django-admin startproject' using Django 5.1.8.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.1/ref/settings/
"""

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-6q-c*qtbkr+j(t-ny#%3c@(4c)_y0-xqz_!y&!1d-m0jn+qkb6"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []


# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "db_chat",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "db_chat_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "db_chat_project.wsgi.application"


# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

STATIC_URL = "static/"

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Database Access Configuration
ALLOWED_MODELS = []  # Specify models in "app_label.ModelName" format

# LLM Configuration
LLM_PROVIDER = "anthropic"  # or "openai", "google", etc.
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"
# Add other LLM keys/configs as needed
# OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# Conversation Persistence Settings
# Options:
# - 'redis': Use Redis for conversation storage (requires 'redis' package)
# - 'memory': Use in-memory storage (no persistence between restarts)
CONVERSATION_STORAGE_TYPE = os.environ.get("CONVERSATION_STORAGE_TYPE", "redis")

# Redis connection URL (only used if CONVERSATION_STORAGE_TYPE is 'redis')
# Format: redis://username:password@host:port/db
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# How long conversations should be kept (in seconds)
# Default: 1 week (60 * 60 * 24 * 7)
CONVERSATION_TTL_SECONDS = int(
    os.environ.get("CONVERSATION_TTL_SECONDS", 60 * 60 * 24 * 7)
)

# Maximum number of previous messages to include in conversation context
# Default: 10 messages
CONVERSATION_CONTEXT_LIMIT = int(os.environ.get("CONVERSATION_CONTEXT_LIMIT", 10))


DB_CHAT = {
    "CLARIFICATION_ENABLED": True,
    "SUMMARIZATION_ENABLED": True,
    "POST_PROCESSOR_ENABLED": True,
    "ENABLE_SCHEMA_CACHING": True,
    "SCHEMA_CACHE_TTL": 300,
}
