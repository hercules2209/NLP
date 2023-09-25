# Sentiment Analysis on Incoming Calls at Helpdesk

## Introduction
This project aims to perform sentiment analysis on incoming calls at a helpdesk. It uses a combination of machine learning models and various libraries to transcribe, translate, and classify the sentiment of spoken conversations. This code was made for SIH hackathon 2023 problem statement number 1356.

## Dependencies
Before running the code, make sure you have the following dependencies installed:
- Python 3.x
- git
- PyTorch
- Transformers
- Whisper
- Pyannote
- scikit-learn
- pandas
- matplotlib
- Twilio
- ffmpeg (for audio processing)

Run dependencies.py to install dependecies
You can do this by running the following command in your terminal
```bash
python dependencies.py
```

You can also install most of these dependencies using pip:
```bash
pip install torch transformers whisper pyannote scikit-learn pandas matplotlib twilio
pip install -q git+https://github.com/openai/whisper.git > /dev/null
pip install -q git+https://github.com/pyannote/pyannote-audio > /dev/null
```

## Models
### Speaker Embedding Model
- Used for speaker verification and clustering.
- Model: "speechbrain/spkrec-ecapa-voxceleb"
- Hugging face link: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb

### Transcription Model
- Used to transcribe spoken conversations.
- Model: Whisper ASR (Automatic Speech Recognition)
- References used
  	https://github.com/openai/whisper </br>
 	https://github.com/m-bain/whisperX/blob/main/README.md
 	
### Sentiment Classification Model
- Used for classifying the sentiment of transcribed text.
- Model: "j-hartmann/emotion-english-distilroberta-base"
- Citation: Jochen Hartmann, "Emotion English DistilRoBERTa-base". https://huggingface.co/j-hartmann/emotion-english-distilroberta-base/, 2022.
- Hugging face link: https://huggingface.co/j-hartmann/emotion-english-distilroberta-base

### Translation Model
- Used for translating text from one language to another.
- Model: "facebook/mbart-large-50-many-to-many-mmt"
- Hugging face link: https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt
- Citation: @article{tang2020multilingual,
    title={Multilingual Translation with Extensible Multilingual Pretraining and Finetuning},
    author={Yuqing Tang and Chau Tran and Xian Li and Peng-Jen Chen and Naman Goyal and Vishrav Chaudhary and Jiatao Gu and Angela Fan},
    year={2020},
    eprint={2008.00401},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

## Code Structure
- `Main Model Pipeline`: Contains the main code for transcription, speaker clustering, translation, and sentiment analysis.
- `Analysis Class`: Contains methods for transcription, translation, and sentiment classification.
- `process_input()`: Transcribes, translates, and classifies incoming call data.
- `VOIP Call`: Initiates a VOIP call using Twilio.

## Usage in jupyter notebook
1. Install the required dependencies as mentioned above.
2. Set your Twilio account SID and authentication token in the code.
3. Run the `process_input()` function with the path to your audio file to perform sentiment analysis.

## Example
```python
# Process an incoming call recording
path_to_audio = 'path/to/your/audio.wav'
result_df = process_input(path_to_audio)
print(result_df)
```
## VOIP Call
You can also use this project to initiate VOIP calls using Twilio. Make sure to set up your Twilio account credentials and specify the number you want to call.

## Usage with `analysis.py`

To use the functionality provided by this project, you can import the `Analysis` class from the `analysis.py` module into your Python script. Here's how you can do it:

```python
from analysis import Analysis
```

Next, you can create an instance of the `Analysis` class:

```python
analyzer = Analysis()
```

Now, you have an `analyzer` object that you can use to perform various tasks.

### Transcribing an Incoming Call

To transcribe an incoming call recording, use the `transcribe` method by providing the path to the audio file as an argument:

```python
audio_file_path = 'path/to/your/audio.wav'
transcribed_text = analyzer.transcribe(audio_file_path, num_speakers)
print(transcribed_text)
```

The `num_speakers` parameter specifies the number of speakers to cluster in the conversation.

### Translating Text

Once you have transcribed text, you can use the `translate` method to translate it into another language. Pass the transcribed text as an argument:

```python
translated_text = analyzer.translate(transcribed_text)
print(translated_text)
```

### Classifying Sentiment

To classify the sentiment of the translated text, use the `classify` method. Provide the translated text as an argument:

```python
sentiment_result = analyzer.classify(translated_text)
print(sentiment_result)
```

### VOIP Call (Twilio Integration)

Additionally, this project allows you to initiate VOIP calls using Twilio. Before using this feature, set up your Twilio account credentials and specify the number you want to call.

## Example
Here is an example of a complete workflow:

```python
from analysis import Analysis

# Create an instance of the Analysis class
analyzer = Analysis()

# Transcribe the audio from an incoming call
audio_file_path = 'path/to/your/audio.wav'
transcribed_text = analyzer.transcribe(audio_file_path, num_speakers)

# Translate the transcribed text
translated_text = analyzer.translate(transcribed_text)

# Classify sentiment
sentiment_result = analyzer.classify(translated_text)

# Print the sentiment result
print(sentiment_result)
```




