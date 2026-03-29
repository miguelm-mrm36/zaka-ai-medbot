# [Zaka AI - ML Track] Challenge 7.2 - Deployable Medbot

![MedBot icon](static/icon.png)

MedBot is a lightweight medical question-answering web app. This project deploys the previously developed MedBot model through a minimal Flask interface using docker.

This deployment version uses:
- **Flask** for the web application
- **Docker** for containerization
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** as the base model
- **A fine-tuned LoRA adapter** saved from training and loaded at runtime

## What the model does

The app allows a user to type a general medical question into a chat-style interface and receive a short educational answer. It is intended for demonstration and learning purposes only.

Examples of questions:
- “What are common symptoms of diabetes?”
- “What can cause a sore throat?”
- “When should I see a doctor for a fever?”

## How the app works

When the service starts, it:
1. loads the TinyLlama base model,
2. loads the saved LoRA adapter from `model_adapter/`,
3. merges the adapter for inference,
4. keeps the model in memory,
5. exposes a chat interface through Flask.

The frontend provides:
- a clean chat-style interface,
- keyboard submit shortcuts using **Cmd+Enter** on macOS and **Ctrl+Enter** on Windows/Linux,
- a loading indicator while the answer is being generated.

## Setup instructions for running locally via Docker

### 1. Build the Docker image

```bash
docker build -t medbot .
```

### 2. Run the container

```bash
docker run -p 9978:9978 medbot
```

### 3. Open the app

Visit:

```text
http://localhost:9978
```

## How to use the interface

1. Open the app in the browser.
2. Type a medical question into the input box.
3. Press **Send** or use **Cmd+Enter / Ctrl+Enter**.
4. Wait for the loading spinner.
5. Read the generated answer in the chat window.

## Deployment notes

This repository includes a `Dockerfile` and a `render.yaml` file to support Docker-based deployment. The app is intentionally packaged so the same codebase can be run locally and hosted online.

## Known issues or limitations

- Startup can be slow because the model is loaded once when the app launches.
- CPU inference is slower than GPU inference.
- This is an educational chatbot and should not be used for diagnosis, prescriptions, or emergencies.
- The model can still make mistakes, so medical advice should always be verified with a qualified professional.

## Preview

