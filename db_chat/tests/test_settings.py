SECRET_KEY = "test"
DEBUG = True
ALLOWED_HOSTS = ["*"]
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "db_chat",
]
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}
ROOT_URLCONF = "db_chat.tests.test_urls"
MIDDLEWARE = []
USE_TZ = True
