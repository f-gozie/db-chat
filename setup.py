from setuptools import setup, find_packages
import os

# Function to read requirements
def read_requirements(filename="requirements.txt"):
    with open(os.path.join(os.path.dirname(__file__), filename), "r") as f:
        return f.read().splitlines()

# Function to read the README
def read_readme(filename="README.md"):
    with open(os.path.join(os.path.dirname(__file__), filename), "r", encoding="utf-8") as f:
        return f.read()

setup(
    name="django-db-chat",
    version="0.1.0",
    packages=find_packages(include=['db_chat', 'db_chat.*']),
    include_package_data=True, # Include other files like templates/static if added later
    install_requires=read_requirements(), # Get dependencies from requirements.txt
    description="A Django app providing a natural language interface to query databases directly via LLMs.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Favour", # Replace with your name/handle
    author_email="your_email@example.com", # Optional: Replace with your email
    url="<your_repo_url_here>", # Optional: Link to your Git repo if you have one
    classifiers=[
        "Environment :: Web Environment",
        "Framework :: Django",
        "Framework :: Django :: 3.2", # Specify compatible Django versions
        "Framework :: Django :: 4.0",
        "Framework :: Django :: 5.0",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Database",
        "Topic :: Communications :: Chat",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
) 