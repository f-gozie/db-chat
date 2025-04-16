# Django DB Chat

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Django](https://img.shields.io/badge/Django-3.2%2B-green.svg)](https://www.djangoproject.com/)

A powerful, modular Django application that translates natural language questions into SQL queries, allowing users to interact with your database using plain English. Powered by Large Language Models and built with a clean, extensible architecture.

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=DB+Chat+Demo" alt="DB Chat Demo" width="600">
</p>

## ✨ Key Features

* **Natural Language → SQL**: Transform plain English questions into precise SQL queries
* **Smart Schema Discovery**: Automatically extracts schema and constraints from Django models
* **Relationship Awareness**: Understands model relationships and dependencies
* **Handles Enum Choices**: Correctly manages field choices with exact capitalization and spacing
* **Multi-LLM Support**: Works with both Anthropic Claude and OpenAI models
* **Conversation Context**: Maintains conversation history for follow-up questions
* **Clean Architecture**: Modular design makes extension and customization simple
* **API-First Design**: Easy to integrate into any Django application

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Django 3.2+
* PostgreSQL database
* API key for either Anthropic or OpenAI (or both)

### Installation

1. Install the package:
   ```bash
   pip install django-db-chat  # Coming soon to PyPI
   # Or from source
   pip install -e .
   ```

2. Add to your Django `INSTALLED_APPS`:
   ```python
   INSTALLED_APPS = [
       # ...
       'db_chat',
   ]
   ```

3. Configure your settings:
   ```python
   # Choose your LLM provider
   LLM_PROVIDER = "anthropic"  # or "openai"
   
   # API keys
   ANTHROPIC_API_KEY = "your_api_key_here"  # If using Anthropic
   OPENAI_API_KEY = "your_api_key_here"     # If using OpenAI
   
   # Specify which models to expose to the chatbot
   ALLOWED_MODELS = [
       "users.User",
       "projects.Project",
       # Add more models as needed
   ]
   ```

4. Include the URLs:
   ```python
   urlpatterns = [
       # ...
       path('api/chat/', include('db_chat.urls')),
   ]
   ```

5. Run migrations and server:
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

### Usage

Send a POST request to `/api/chat/query/` with:

```json
{
  "query": "How many users are in the system?",
  "conversation_id": "optional-existing-conversation-id"
}
```

Response:

```json
{
  "reply": "There are 1,234 users in the system.",
  "conversation_id": "conversation-uuid"
}
```

## 🏗️ Architecture

The application follows a clean, modular architecture:

### Core Components

1. **Model Registry**: Automatically discovers model schemas and relationships
   - Extracts field types, constraints, and choices from Django models
   - Builds dependency graph for related models

2. **LLM Adapter**: Provides a unified interface to different LLM providers
   - Adapter pattern with implementations for Anthropic and OpenAI
   - Factory method for selecting the appropriate adapter

3. **Prompt Templates**: Cleanly separated prompt templates for different tasks
   - SQL generation prompts
   - Result interpretation prompts

4. **Chat Service**: Core business logic
   - Handles the full query lifecycle
   - Manages conversation history

5. **Storage Adapters**: Flexible storage for conversation history
   - Redis-based persistence
   - In-memory option for development

### Diagnostic Endpoint

View detected models and their relationships at:
```
/api/chat/schema-info/
```

This shows:
- All detected models and their table names
- Fields with choices and their exact values
- Dependency relationships between models

## 🧩 Extending the Application

### Adding New LLM Providers

Extend the `LLMAdapter` class:

```python
class NewProviderAdapter(LLMAdapter):
    def __init__(self):
        # Initialize client for the new provider
        pass
        
    def generate_text(self, system_prompt, messages, max_tokens=1024):
        # Implement text generation using the new provider
        pass
```

Then update the `get_adapter` method in `LLMAdapter`.

### Adding New SQL Validation Rules

Add new validation functions to the `utils.py` module following the pattern of existing validators.

### Customizing Prompts

Edit the templates in `prompts.py` to customize how the chatbot interacts with the LLM.

## 📝 License

MIT License - See LICENSE file for details