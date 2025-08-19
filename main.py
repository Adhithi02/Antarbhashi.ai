import os
import time
import wave
import torch
import whisper
import pyaudio
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
from pydub import AudioSegment
from parler_tts import ParlerTTSForConditionalGeneration
import soundfile as sf



class SpeechTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("English to Indian Language Translator")
        self.root.geometry("600x550")  
        self.root.resizable(True, True)
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 12))
        self.style.configure("TLabel", font=("Arial", 12))
        
        # Create app directory
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.audio_dir = os.path.join(self.app_dir, "audio_files")
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Language mapping
        self.language_mapping = {
        "Kannada": {
            "code": "kan_Knda",
            "speakers": ["Suresh", "Anu", "Chetan", "Vidya"],
            "default_speaker": "Suresh"
        },
        "Telugu": {
            "code": "tel_Telu",
            "speakers": ["Prakash", "Lalitha", "Kiran"],
            "default_speaker": "Prakash"
        },
        "Hindi": {
            "code": "hin_Deva",
            "speakers": ["Ravi", "Priya", "Amit"],
            "default_speaker": "Ravi"
        },
        "Tamil": {
            "code": "tam_Taml",
            "speakers": ["Arun", "Meena"],
            "default_speaker": "Arun"
        },
        "Gujarati": {
            "code": "guj_Gujr",
            "speakers": ["Jignesh", "Kavita"],
            "default_speaker": "Jignesh"
        },
        "Bengali": {
            "code": "ben_Beng",
            "speakers": ["Rahul", "Mou"],
            "default_speaker": "Rahul"
        }
    }

        
        # Recording variables
        self.is_recording = False
        self.recorded_audio = []
        self.audio_thread = None
        self.temp_wav_file = os.path.join(self.audio_dir, "temp_recording.wav")
        self.output_file = os.path.join(self.audio_dir, "output.mp3")
        
        # Initialize models
        self.load_models_thread = threading.Thread(target=self.load_models)
        self.load_models_thread.daemon = True
        self.load_models_thread.start()
        
        # Create UI
        self.create_widgets()
    
    def load_models(self):
        """Load all models in a background thread to keep UI responsive"""
        try:
            # Update status
            self.update_status("Loading models, please wait...")
            
            # Device configuration
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load Whisper model
            self.update_status("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            
            # Load translation model
            self.update_status("Loading IndicTrans model...")
            model_name = "ai4bharat/indictrans2-en-indic-1B"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
            self.ip = IndicProcessor(inference=True)
            
            # Load TTS model
            self.update_status("Loading Indic Parler-TTS model...")
            self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(self.DEVICE)
            self.tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
            self.description_tokenizer = AutoTokenizer.from_pretrained(self.tts_model.config.text_encoder._name_or_path)
            
            # Optimize TTS model for inference
            self.tts_model.eval()
            
            # Enable model optimizations for faster inference
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                try:
                    self.tts_model = torch.compile(self.tts_model, mode="reduce-overhead")
                    print("TTS model compiled with torch.compile for faster inference")
                except Exception as e:
                    print(f"torch.compile failed, using standard model: {e}")
            
            # Enable half precision for faster inference if using CUDA
            if self.DEVICE == "cuda" and hasattr(self.tts_model, 'half'):
                try:
                    self.tts_model = self.tts_model.half()
                    print("TTS model converted to half precision for faster inference")
                except Exception as e:
                    print(f"Half precision conversion failed: {e}")
            
            # Enable memory efficient attention if available
            if hasattr(self.tts_model.config, 'use_memory_efficient_attention'):
                self.tts_model.config.use_memory_efficient_attention = True
            
            # Pre-allocate tensors for faster inference
            self.tts_model.config.pre_allocate_tensors = True
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.tts_model, 'gradient_checkpointing_enable'):
                self.tts_model.gradient_checkpointing_enable()
            
            # Cache tokenizer outputs for common descriptions
            self.speaker_descriptions = {}
            
            self.update_status("All models loaded! Ready to record.")
            self.record_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.update_status(f"Error loading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="English to Indian Language Translator", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        self.status_label = ttk.Label(main_frame, text="Loading models, please wait...", foreground="blue")
        self.status_label.pack(pady=10)
        
        lang_frame = ttk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(lang_frame, text="Target Language:").pack(side=tk.LEFT, padx=5)
        
        self.language_var = tk.StringVar(value="Kannada")
        languages = list(self.language_mapping.keys())
        language_combo = ttk.Combobox(lang_frame, textvariable=self.language_var, values=languages, width=15)
        language_combo.pack(side=tk.LEFT, padx=5)
        language_combo.bind("<<ComboboxSelected>>", self.on_language_change)
        
        self.record_button = ttk.Button(main_frame, text="Start Recording", command=self.toggle_recording, state=tk.DISABLED)
        self.record_button.pack(pady=15)
        
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=400, mode="indeterminate")
        self.progress.pack(pady=10)
        text_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(text_frame, text="English:").pack(anchor=tk.W)
        self.english_text = tk.Text(text_frame, height=4, wrap=tk.WORD)
        self.english_text.pack(fill=tk.X, pady=5)
        
        self.translated_label = ttk.Label(text_frame, text="Kannada:")
        self.translated_label.pack(anchor=tk.W)
        self.translated_text = tk.Text(text_frame, height=4, wrap=tk.WORD)
        self.translated_text.pack(fill=tk.X, pady=5)
        
        speaker_frame = ttk.Frame(main_frame)
        speaker_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(speaker_frame, text="Speaker:").pack(side=tk.LEFT, padx=5)
        self.speaker_var = tk.StringVar(value=self.language_mapping["Kannada"]["default_speaker"])
        self.speaker_combo = ttk.Combobox(speaker_frame, textvariable=self.speaker_var, 
                                     values=self.language_mapping["Kannada"]["speakers"], width=15)
        self.speaker_combo.pack(side=tk.LEFT, padx=5)
        
        # Add speed/quality toggle
        speed_frame = ttk.Frame(main_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(speed_frame, text="Generation Speed:").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.StringVar(value="Balanced")
        speed_combo = ttk.Combobox(speed_frame, textvariable=self.speed_var, 
                                  values=["Fast", "Balanced", "High Quality"], width=15)
        speed_combo.pack(side=tk.LEFT, padx=5)
        
        self.play_button = ttk.Button(main_frame, text="Play Translation", command=self.play_audio, state=tk.DISABLED)
        self.play_button.pack(pady=10)
    
    def on_language_change(self, event=None):
        selected_language = self.language_var.get()
        if selected_language in self.language_mapping:
            speakers = self.language_mapping[selected_language]["speakers"]
            default_speaker = self.language_mapping[selected_language]["default_speaker"]
            
            self.speaker_combo['values'] = speakers
            self.speaker_var.set(default_speaker)
            
            self.translated_label.config(text=f"{selected_language}:")
    
    def update_status(self, message):
        def _update():
            self.status_label.config(text=message)
        
        if threading.current_thread() is threading.main_thread():
            _update()
        else:
            self.root.after(0, _update)
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Hell yea bro mic testing 1.2.3.4.5 ahhahahahah screw this shit i wanna sleep"""
        self.is_recording = True
        self.recorded_audio = []
        self.record_button.config(text="Stop Recording")
        self.update_status("Recording... Speak now")
        
        self.english_text.delete(1.0, tk.END)
        self.translated_text.delete(1.0, tk.END)
        self.play_button.config(state=tk.DISABLED)
        self.audio_thread = threading.Thread(target=self.record_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
    
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.record_button.config(text="Start Recording")
            self.update_status("Processing audio...")
            
            # Wait for recording thread to finish this is the important part to you all for tomm cuz you need to time this shit for the shit ahhahaha
            if self.audio_thread:
                self.audio_thread.join()
            
            
            processing_thread = threading.Thread(target=self.process_recording)
            processing_thread.daemon = True
            processing_thread.start()
    
    def record_audio(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        try:
            p = pyaudio.PyAudio()
            info = f"Audio devices available:\n"
            for i in range(p.get_device_count()):
                dev_info = p.get_device_info_by_index(i)
                info += f"Device {i}: {dev_info['name']}, Inputs: {dev_info['maxInputChannels']}\n"
            print(info)
            
            default_input_device_index = p.get_default_input_device_info()['index']
            print(f"Using default input device: {default_input_device_index}")
            
            stream = p.open(format=FORMAT,
                           channels=CHANNELS,
                           rate=RATE,
                           input=True,
                           input_device_index=default_input_device_index,
                           frames_per_buffer=CHUNK)
            
            self.update_status("Recording... Speak now")
            
            while self.is_recording:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    self.recorded_audio.append(data)
                except Exception as e:
                    print(f"Error during recording: {e}")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            if len(self.recorded_audio) > 0:
                print(f"Saving {len(self.recorded_audio)} audio chunks to {self.temp_wav_file}")
                
                os.makedirs(os.path.dirname(self.temp_wav_file), exist_ok=True)
                
                wf = wave.open(self.temp_wav_file, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(self.recorded_audio))
                wf.close()
                
                if os.path.exists(self.temp_wav_file):
                    file_size = os.path.getsize(self.temp_wav_file)
                    print(f"Recording saved successfully. File size: {file_size} bytes")
                    if file_size < 100:
                        print("WARNING: Audio file is very small. Microphone might not be capturing audio.")
                else:
                    print(f"ERROR: Failed to create audio file at {self.temp_wav_file}")
            else:
                print("ERROR: No audio data recorded!")
                self.update_status("No audio recorded. Please check your microphone.")
        
        except Exception as e:
            print(f"Critical error in recording: {str(e)}")
            self.update_status(f"Recording error: {str(e)}")
    
    def process_recording(self):
        """Process the recorded audio through the translation pipeline"""
        try:
            # Check if recording file exists and has content
            if not os.path.exists(self.temp_wav_file) or os.path.getsize(self.temp_wav_file) < 100:
                self.update_status("Error: Recording file is empty or too small")
                messagebox.showerror("Error", "No audio was recorded. Please check your microphone settings.")
                return
                
            # Start progress bar
            self.root.after(0, self.progress.start)
            
            # Get the selected language code
            selected_language = self.language_var.get()
            target_lang_code = self.language_mapping[selected_language]["code"]
            
            # Step 1: Transcribe
            self.update_status("Transcribing audio...")
            result = self.whisper_model.transcribe(self.temp_wav_file)
            english_text = result["text"]
            
            if not english_text.strip():
                self.update_status("Error: Could not transcribe any text")
                messagebox.showerror("Error", "No speech detected in the recording. Please try again.")
                self.root.after(0, self.progress.stop)
                return
            
            def update_english():
                self.english_text.delete(1.0, tk.END)
                self.english_text.insert(tk.END, english_text)
            self.root.after(0, update_english)
            
            # Step 2: Translate
            self.update_status(f"Translating to {selected_language}...")
            batch = self.ip.preprocess_batch([english_text], src_lang="eng_Latn", tgt_lang=target_lang_code)
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.DEVICE)
            
            with torch.no_grad():
                output_tokens = self.model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )
            
            with self.tokenizer.as_target_tokenizer():
                decoded = self.tokenizer.batch_decode(
                    output_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            
            translated_text = self.ip.postprocess_batch(decoded, lang=target_lang_code)[0]
            
            def update_translated():
                self.translated_text.delete(1.0, tk.END)
                self.translated_text.insert(tk.END, translated_text)
            self.root.after(0, update_translated)
            
            # Step 3: Text to Speech
            self.update_status(f"Generating {selected_language} speech...")
            speaker = self.speaker_var.get()
            description = f"{speaker}'s voice is clear and natural."
            
            description_inputs = self.description_tokenizer(description, return_tensors="pt")
            prompt_inputs = self.tts_tokenizer(translated_text, return_tensors="pt")
            
            description_input_ids = description_inputs.input_ids.to(self.DEVICE)
            description_attention_mask = description_inputs.attention_mask.to(self.DEVICE)
            prompt_input_ids = prompt_inputs.input_ids.to(self.DEVICE)
            prompt_attention_mask = prompt_inputs.attention_mask.to(self.DEVICE)
            
            with torch.no_grad():
                # Get speed setting and configure generation parameters
                speed_setting = self.speed_var.get()
                
                if speed_setting == "Fast":
                    # Fastest generation with minimal quality trade-offs
                    generation_params = {
                        "max_length": 256,  # Shorter max length
                        "do_sample": True,
                        "temperature": 0.5,  # Lower temperature for faster convergence
                        "top_p": 0.8,
                        "top_k": 20,
                        "num_beams": 1,
                        "early_stopping": True,
                        "pad_token_id": self.tts_tokenizer.eos_token_id,
                        "eos_token_id": self.tts_tokenizer.eos_token_id,
                        "use_cache": True,
                    }
                elif speed_setting == "High Quality":
                    # Highest quality with slower generation
                    generation_params = {
                        "max_length": 1024,  # Longer max length for better quality
                        "do_sample": True,
                        "temperature": 0.8,  # Higher temperature for more natural speech
                        "top_p": 0.95,
                        "top_k": 100,
                        "num_beams": 3,  # Beam search for better quality
                        "early_stopping": True,
                        "pad_token_id": self.tts_tokenizer.eos_token_id,
                        "eos_token_id": self.tts_tokenizer.eos_token_id,
                        "use_cache": True,
                    }
                else:  # Balanced
                    # Balanced speed and quality
                    generation_params = {
                        "max_length": 512,
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 50,
                        "num_beams": 1,
                        "early_stopping": True,
                        "pad_token_id": self.tts_tokenizer.eos_token_id,
                        "eos_token_id": self.tts_tokenizer.eos_token_id,
                        "use_cache": True,
                    }
                
                generation = self.tts_model.generate(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    **generation_params
                )
            
            audio_arr = generation.cpu().numpy().squeeze()
            wav_output = self.output_file.replace(".mp3", ".wav")
            sf.write(wav_output, audio_arr, self.tts_model.config.sampling_rate)
            
            if self.output_file.endswith(".mp3"):
                sound = AudioSegment.from_wav(wav_output)
                sound.export(self.output_file, format="mp3")
                os.remove(wav_output)
            
            # Enable play button
            def enable_play():
                self.play_button.config(state=tk.NORMAL)
                self.progress.stop()
                self.update_status("Translation complete! Ready to play.")
            self.root.after(0, enable_play)
            
        except Exception as e:
            self.root.after(0, self.progress.stop)
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
    
    def play_audio(self):
        try:
            os.system(f"start {self.output_file}" if os.name == "nt" else f"mpg123 {self.output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not play audio: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechTranslatorApp(root)
    root.mainloop()