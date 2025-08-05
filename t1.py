import os
import numpy as np
import sounddevice as sd
import deepspeech
from scipy.io.wavfile import write
import googletrans
from googletrans import Translator, LANGUAGES
from gtts import gTTS
import io
import pygame
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QTextEdit, QLabel, QComboBox, QListWidget
)
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont

# DeepSpeech Model Paths
MODEL_PATH = "deepspeech-0.9.3-models.pbmm"
SCORER_PATH = "deepspeech-0.9.3-models.scorer"
SAMPLE_RATE = 16000

# Load DeepSpeech Model
ds_model = deepspeech.Model(MODEL_PATH)
if os.path.exists(SCORER_PATH):
    ds_model.enableExternalScorer(SCORER_PATH)

# Speech-to-Text Thread
class SpeechToTextThread(QThread):
    text_ready = pyqtSignal(str)

    def run(self):
        self.text_ready.emit("Listening... Speak now!")

        try:
            duration = 5  # Seconds
            recording = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
            sd.wait()

            # Save recording (for debugging)
            write("test_audio.wav", SAMPLE_RATE, recording)

            # Process with DeepSpeech
            text = ds_model.stt(np.frombuffer(recording, dtype=np.int16))
            self.text_ready.emit(text if text else "Could not understand speech.")

        except Exception as e:
            self.text_ready.emit(f"Error: {e}")

# Translation Thread
class TranslationThread(QThread):
    translation_done = pyqtSignal(str)

    def __init__(self, text, target_lang):
        super().__init__()
        self.text = text
        self.target_lang = target_lang

    def run(self):
        try:
            translator = Translator()
            translated = translator.translate(self.text, dest=self.target_lang)
            self.translation_done.emit(translated.text)
        except Exception as e:
            self.translation_done.emit(f"Translation Error: {str(e)}")

# Translator App
class TranslatorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.translation_history = []
        self.speechThread = SpeechToTextThread()
        self.speechThread.text_ready.connect(self.displaySpeechText)

    def initUI(self):
        self.setWindowTitle('AI Voice Translator')
        self.resize(600, 500)
        layout = QVBoxLayout()

        self.label = QLabel("Enter text or use voice input:")
        layout.addWidget(self.label)

        self.inputText = QLineEdit()
        self.inputText.setPlaceholderText("Type or speak...")
        self.inputText.setFont(QFont("Arial", 14))
        layout.addWidget(self.inputText)

        self.voiceButton = QPushButton("🎤 Speak")
        self.voiceButton.clicked.connect(self.startSpeechRecognition)
        layout.addWidget(self.voiceButton)

        self.speakMessage = QLabel("")
        self.speakMessage.setFont(QFont("Arial", 12))
        layout.addWidget(self.speakMessage)

        self.languageDropdown = QComboBox()
        self.languages = {f"{LANGUAGES[lang].title()} ({lang})": lang for lang in LANGUAGES}
        self.languageDropdown.addItems(self.languages.keys())
        layout.addWidget(self.languageDropdown)

        self.translateButton = QPushButton("Translate")
        self.translateButton.clicked.connect(self.startTranslationThread)
        layout.addWidget(self.translateButton)

        self.outputText = QTextEdit()
        self.outputText.setReadOnly(True)
        self.outputText.setFont(QFont("Arial", 16))
        layout.addWidget(self.outputText)

        self.speakButton = QPushButton("🔊 Speak Translation")
        self.speakButton.clicked.connect(self.speakTranslation)
        layout.addWidget(self.speakButton)

        self.historyLabel = QLabel("Translation History:")
        layout.addWidget(self.historyLabel)

        self.historyList = QListWidget()
        self.historyList.setFont(QFont("Arial", 12))
        layout.addWidget(self.historyList)

        self.setLayout(layout)

    def startSpeechRecognition(self):
        self.speakMessage.setText("Listening... Speak now!")
        self.speechThread.start()

    def displaySpeechText(self, text):
        if text == "Listening... Speak now!":
            self.speakMessage.setText(text)
        else:
            self.speakMessage.setText("")
            self.inputText.setText(text)
            self.startTranslationThread()

    def startTranslationThread(self):
        text = self.inputText.text()
        if text:
            selected_lang = self.languages[self.languageDropdown.currentText()]
            self.translationThread = TranslationThread(text, selected_lang)
            self.translationThread.translation_done.connect(self.displayTranslation)
            self.translationThread.start()
        else:
            self.outputText.setText("Please enter or speak text!")

    def displayTranslation(self, translated_text):
        self.outputText.setText(translated_text)
        self.speakTranslation()

        original_text = self.inputText.text()
        history_entry = f"{original_text} ➝ {translated_text}"
        self.translation_history.append(history_entry)
        self.historyList.addItem(history_entry)

    def speakTranslation(self):
        text = self.outputText.toPlainText()
        if text:
            selected_lang = self.languages[self.languageDropdown.currentText()]
            tts = gTTS(text=text, lang=selected_lang, slow=False)

            # Play without saving MP3 file
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)

            pygame.mixer.init()
            pygame.mixer.music.load(fp)
            pygame.mixer.music.play()

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print("DeepSpeech model not found!")
        exit(1)

    app = QApplication([])
    translatorApp = TranslatorApp()
    translatorApp.show()
    app.exec_()
