import whisper
import datetime
import subprocess
import torch
import torch.nn.functional as F
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
import wave
import contextlib
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from transformers import pipeline
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda"))

def segment_embedding(segment,audio,duration,path):
  start = segment["start"]

  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = audio.crop(path, clip)


  waveform = waveform.mean(dim=0, keepdim=True)
  return embedding_model(waveform.unsqueeze(0))

def time(secs):
  return datetime.timedelta(seconds=round(secs))



class Analysis:

    def __init__(self):
        self.model_transcribe = whisper.load_model('medium')
        self.classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",tokenizer='j-hartmann/emotion-english-distilroberta-base',)#top_k=None)
        self.translate_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.translate_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")


    def transcribe(self, path,num_speakers):
        if path[-3:] != 'wav':
          subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
          path = 'audio.wav'
        result = self.model_transcribe.transcribe(path)
        segments = result["segments"]

        with contextlib.closing(wave.open(path,'r')) as f:
          frames = f.getnframes()
          rate = f.getframerate()
          duration = frames / float(rate)

        audio = Audio()

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
          embeddings[i] = segment_embedding(segment,audio,duration,path)

        embeddings = np.nan_to_num(embeddings)

        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
          segments[i]["speaker"] = str(labels[i] + 1)

        f = open("transcript.txt", "w",encoding="UTF-8")

        for (i, segment) in enumerate(segments):
          if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            f.write("\n")
          f.write(segment["text"][1:] + ' ')
        f.close()

        transcription_data=open('transcript.txt','r',encoding="UTF-8").read()
        print(transcription_data)
        return transcription_data


    def translate(self,text):
      self.translate_tokenizer.src_lang = "hi_IN"
      encoded_hi = self.translate_tokenizer(text, return_tensors="pt")
      generated_tokens = self.translate_model.generate(**encoded_hi, forced_bos_token_id=self.translate_tokenizer.lang_code_to_id["en_XX"])
      a = self.translate_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
      print(a)

      return a


    def classify(self, sentences):
      return self.classifier(sentences)