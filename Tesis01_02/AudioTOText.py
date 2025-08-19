import whisper

model = whisper.load_model("base")  # Puedes probar con "small", "medium", o "large"
result = model.transcribe("Banco.ogg")
print(result["text"])
