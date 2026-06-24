# File-Aware AI Chatbot

A local Python web chatbot that answers questions about uploaded user data and documents using NVIDIA's OpenAI-compatible API.

## Important pip note

Do not install these with pip:

```text
cgi json html traceback http uuid
```

They are built into Python's standard library. Installing packages with those names can fail or shadow the real built-in modules.

If you installed the third-party `uuid` package, remove it:

```powershell
pip uninstall uuid
```

Install only the external packages this app needs:

```powershell
pip install -r requirements.txt
```

## Run

```powershell
$env:NVIDIA_API_KEY="your_key_here"
python app.py
```

Then open:

```text
http://127.0.0.1:8501
```

You can also leave `NVIDIA_API_KEY` unset and paste the API key into the web UI. The app does not save the key in source files.

## Model

```text
meta/llama-3.1-8b-instruct
```

## Supported uploads

- CSV
- Excel `.xlsx`
- PDF
- Word `.docx`
- Text-like files such as `.txt`, `.json`, `.log`, `.md`
- Other file types are accepted, but binary formats may only expose basic metadata.
