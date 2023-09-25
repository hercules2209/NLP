import subprocess


pip_commands = [
    "pip install transformers",
    "pip install torch torchaudio torchvision pandas",
    "pip install -q git+https://github.com/openai/whisper.git > /dev/null",
    "pip install -q git+https://github.com/pyannote/pyannote-audio > /dev/null",
    "pip install sentencepiece",
    "pip freeze | grep transformers",
    "pip install pandas"
    "pip install matplotlib"
    "pip install twilio"
]


for command in pip_commands:
    subprocess.run(command, shell=True)

