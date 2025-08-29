# LINTING.md

## VS Code Setup for Python Linting and Formatting

This guide provides step-by-step instructions to configure VS Code for optimal Python development with this project's linting and formatting standards.

### Required VS Code Extensions

Install these extensions through the VS Code Extensions marketplace:

1. **Python** (ms-python.python) - Core Python support
2. **Pylint** (ms-python.pylint) - Python linting
3. **Flake8** (ms-python.flake8) - Additional Python linting
4. **Black Formatter** (ms-python.black-formatter) - Python code formatting
5. **isort** (ms-python.isort) - Python import sorting

### Project Configuration

The project already includes VS Code settings in `.vscode/settings.json`.

### Recommended VS Code Settings Update

Update your `.vscode/settings.json` to align with project standards:

```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Args": ["--max-line-length=150"],
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "150"],
  "editor.formatOnSave": true,
  "editor.rulers": [150],
  "python.linting.flake8CategorySeverity.E": "Warning",
  "python.linting.flake8CategorySeverity.W": "Warning",
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.autoImportCompletions": true,
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".venv": false
  },
  "python.terminal.activateEnvironment": true,
  "python.envFile": "${workspaceFolder}/.env",
  "python.analysis.extraPaths": ["./bcnc"],
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### Setup Steps

1. **Install Python Virtual Environment**

   ```bash
   # Activate the virtual environment
   source .venv/bin/activate

   # Install development dependencies
   pip install -r requirements.txt
   ```

2. **Install VS Code Extensions**

   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search and install each extension listed above

3. **Verify Python Interpreter**

   - Open Command Palette (Ctrl+Shift+P)
   - Type "Python: Select Interpreter"
   - Choose `./.venv/bin/python`

4. **Test Configuration**
   - Open a Python file
   - Make some formatting changes
   - Save the file (Ctrl+S) - it should auto-format with Black
   - Check that flake8 warnings appear in the Problems panel

### Project Linting Standards

- **Line Length**: 150 characters
- **Formatter**: Black with isort for import sorting
- **Linter**: Flake8 (Pylint disabled)
- **Ignored Rules**:
  - E203: Whitespace before ':'
  - W503: Line break before binary operator
  - E402: Module level import not at top of file
  - F403: 'from module import \*' used
  - F541: F-string is missing placeholders
  - E722: Do not use bare except
  - E501: Line too long (handled by Black)

### Manual Formatting Commands

From the project root with virtual environment activated:

```bash
# Format code with Black
black . --line-length 150

# Sort imports with isort
isort . --profile black --line-length 150

# Run flake8 linting
flake8 . --max-line-length=150

# Run pylint (if needed)
pylint . --max-line-length=150
```

### Troubleshooting

**Extensions not working:**

- Restart VS Code after installing extensions
- Verify Python interpreter is set to `./.venv/bin/python`
- Check that the virtual environment is activated

**Formatting not applying:**

- Ensure `editor.formatOnSave` is `true`
- Check that Black formatter is set as default formatter for Python files
- Verify no conflicting formatters are enabled

**Linting not showing:**

- Check that flake8 extension is installed and enabled
- Verify Python linting is enabled in settings
- Look for errors in the Output panel (Python, Flake8 channels)

**Import sorting not working:**

- Ensure isort extension is installed
- Check that `source.organizeImports` is enabled in code actions on save
- Verify isort is configured with Black profile
