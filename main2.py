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
import numpy as np
import librosa
import webrtcvad
from scipy.ndimage import uniform_filter1d
import queue


class AudioEnhancer:
    """Real-time audio enhancement for noise reduction - using spectral3.py approach"""
    def __init__(self, samplerate=16000, frame_length=2048, hop_length=512, 
                 alpha=0.98, noise_threshold=1.2, vad_aggressiveness=2):
        self.samplerate = samplerate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.alpha = alpha
        self.noise_threshold = noise_threshold
        
        # Initialize VAD - exactly as in spectral3.py
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.blocksize = 320  # 20 ms frames for VAD at 16kHz (same as spectral3.py)
        
        # Noise profile for enhancement - exactly as in spectral3.py
        self.noise_profile = np.zeros((frame_length // 2 + 1, 1))
        self.noise_frames_count = 0
        self.speech_frames_count = 0
        
    def enhance_audio_chunk(self, audio_chunk):
        """Enhance a chunk of audio using the exact method from spectral3.py"""
        if len(audio_chunk) < self.frame_length:
            return audio_chunk
            
        # changing from time domain to frq domain (same comment as spectral3.py)
        stft = librosa.stft(audio_chunk, n_fft=self.frame_length, hop_length=self.hop_length)
        mag, phase = np.abs(stft), np.angle(stft)
        avg_mag = np.mean(mag, axis=1, keepdims=True)

        # (must be 20ms for webrtcvad) idk why but it works better with 20ms (same comment as spectral3.py)
        vad_frame = audio_chunk[:self.blocksize]  # first 20ms frame exactly
        vad_frame_bytes = (vad_frame * 32768).astype(np.int16).tobytes()
        
        try:
            is_speech = self.vad.is_speech(vad_frame_bytes, self.samplerate)
        except:
            is_speech = True  # Default to speech if VAD fails

        if not is_speech:
            self.noise_profile = self.alpha * self.noise_profile + (1 - self.alpha) * avg_mag
            self.noise_frames_count += 1
        else:
            self.speech_frames_count += 1

        # (changed mask for smooth audio) remove this if its still same ***** (same comment as spectral3.py)
        mask = mag / (mag + self.noise_threshold * self.noise_profile + 1e-8) #**tune**
        mag_enh = mag * mask
        mag_enh = uniform_filter1d(mag_enh, size=3, axis=1)

        # changing it back to time domain (same comment as spectral3.py)
        stft_enh = mag_enh * np.exp(1j * phase)
        enhanced = librosa.istft(stft_enh, hop_length=self.hop_length)

        return enhanced
    
    def enhance_full_audio(self, audio_data):
        """Enhance complete audio file using spectral3.py approach"""
        if len(audio_data) == 0:
            return audio_data
        
        # Reset noise profile for new audio
        self.noise_profile = np.zeros((self.frame_length // 2 + 1, 1))
        self.noise_frames_count = 0
        self.speech_frames_count = 0
        
        # Process the entire audio similar to spectral3.py real-time approach
        # but in larger chunks for file processing
        buffer = np.array(audio_data)
        all_enhanced_audio = []
        
        # Process in frame_length chunks (similar to spectral3.py buffer processing)
        start = 0
        while start < len(buffer):
            end = min(start + self.frame_length, len(buffer))
            chunk = buffer[start:end]
            
            if len(chunk) >= self.frame_length:
                enhanced_chunk = self.enhance_audio_chunk(chunk)
                all_enhanced_audio.append(enhanced_chunk)
            elif len(chunk) > 0:
                # Handle remaining short chunk
                # Pad to minimum length for processing
                padded_chunk = np.pad(chunk, (0, self.frame_length - len(chunk)), mode='constant')
                enhanced_chunk = self.enhance_audio_chunk(padded_chunk)
                # Take only the original length
                enhanced_chunk = enhanced_chunk[:len(chunk)]
                all_enhanced_audio.append(enhanced_chunk)
            
            start += self.frame_length
        
        if all_enhanced_audio:
            # Concatenate all enhanced chunks (same as spectral3.py final_audio)
            final_audio = np.concatenate(all_enhanced_audio)
            # Normalize (same as spectral3.py)
            if np.max(np.abs(final_audio)) > 0:
                final_audio = final_audio / np.max(np.abs(final_audio))
            return final_audio
        
        return audio_data


class SpeechTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced English to Indian Language Translator")
        self.root.geometry("650x650")  
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
        self.enhanced_wav_file = os.path.join(self.audio_dir, "enhanced_recording.wav")
        self.output_file = os.path.join(self.audio_dir, "output.mp3")
        
        # Initialize audio enhancer with spectral3.py parameters
        self.audio_enhancer = AudioEnhancer(
            samplerate=16000,
            frame_length=2048,
            hop_length=512,
            alpha=0.98,
            noise_threshold=1.2,  # Default from spectral3.py
            vad_aggressiveness=2
        )
        
        # Audio enhancement settings
        self.enhancement_enabled = tk.BooleanVar(value=True)
        self.noise_threshold_var = tk.DoubleVar(value=1.2)  # Default from spectral3.py
        self.auto_play_enabled = tk.BooleanVar(value=True)
        
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
            
            self.tts_model.eval()
            
            self.update_status("All models loaded! Ready to record.")
            self.record_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.update_status(f"Error loading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="Enhanced English to Indian Language Translator", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        self.status_label = ttk.Label(main_frame, text="Loading models, please wait...", foreground="blue")
        self.status_label.pack(pady=10)
        
        # Audio Enhancement Settings Frame
        enhancement_frame = ttk.LabelFrame(main_frame, text="Audio Enhancement Settings", padding=10)
        enhancement_frame.pack(fill=tk.X, pady=5)
        
        enhancement_check = ttk.Checkbutton(enhancement_frame, text="Enable Audio Enhancement", 
                                          variable=self.enhancement_enabled)
        enhancement_check.pack(anchor=tk.W)
        
        threshold_frame = ttk.Frame(enhancement_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(threshold_frame, text="Noise Reduction Level:").pack(side=tk.LEFT)
        # Changed range to match spectral3.py default (1.2) with reasonable range
        threshold_scale = ttk.Scale(threshold_frame, from_=0.5, to=3.0, 
                                  variable=self.noise_threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.threshold_label = ttk.Label(threshold_frame, text="1.2")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        threshold_scale.bind("<Motion>", self.update_threshold_label)
        auto_play_check = ttk.Checkbutton(enhancement_frame, text="Auto-play after translation", 
                                        variable=self.auto_play_enabled)
        auto_play_check.pack(anchor=tk.W, pady=5)
        
        # Language Selection Frame
        lang_frame = ttk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(lang_frame, text="Target Language:").pack(side=tk.LEFT, padx=5)
        
        self.language_var = tk.StringVar(value="Kannada")
        languages = list(self.language_mapping.keys())
        language_combo = ttk.Combobox(lang_frame, textvariable=self.language_var, values=languages, width=15)
        language_combo.pack(side=tk.LEFT, padx=5)
        language_combo.bind("<<ComboboxSelected>>", self.on_language_change)
        
        # Recording Button
        self.record_button = ttk.Button(main_frame, text="Start Recording", 
                                       command=self.toggle_recording, state=tk.DISABLED)
        self.record_button.pack(pady=15)
        
        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", length=400, mode="indeterminate")
        self.progress.pack(pady=10)
        
        # Results Frame
        text_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        ttk.Label(text_frame, text="English:").pack(anchor=tk.W)
        self.english_text = tk.Text(text_frame, height=4, wrap=tk.WORD)
        self.english_text.pack(fill=tk.X, pady=5)
        
        self.translated_label = ttk.Label(text_frame, text="Kannada:")
        self.translated_label.pack(anchor=tk.W)
        self.translated_text = tk.Text(text_frame, height=4, wrap=tk.WORD)
        self.translated_text.pack(fill=tk.X, pady=5)
        
        # Speaker Selection Frame
        speaker_frame = ttk.Frame(main_frame)
        speaker_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(speaker_frame, text="Speaker:").pack(side=tk.LEFT, padx=5)
        self.speaker_var = tk.StringVar(value=self.language_mapping["Kannada"]["default_speaker"])
        self.speaker_combo = ttk.Combobox(speaker_frame, textvariable=self.speaker_var, 
                                         values=self.language_mapping["Kannada"]["speakers"], width=15)
        self.speaker_combo.pack(side=tk.LEFT, padx=5)
        
        # Play Button
        self.play_button = ttk.Button(main_frame, text="Play Translation", 
                                     command=self.play_audio, state=tk.DISABLED)
        self.play_button.pack(pady=10)
        
        # Audio Quality Info
        self.quality_label = ttk.Label(main_frame, text="", foreground="green", font=("Arial", 10))
        self.quality_label.pack(pady=5)
    
    def update_threshold_label(self, event=None):
        """Update the threshold label with current value"""
        value = self.noise_threshold_var.get()
        self.threshold_label.config(text=f"{value:.1f}")
        # Update the audio enhancer's threshold
        self.audio_enhancer.noise_threshold = value
    
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
    
    def update_quality_info(self, message):
        """Update audio quality information"""
        def _update():
            self.quality_label.config(text=message)
        
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
        """Start recording with enhanced audio processing"""
        self.is_recording = True
        self.recorded_audio = []
        self.record_button.config(text="Stop Recording")
        self.update_status("Recording... Speak now")
        self.update_quality_info("")
        
        # Reset audio enhancer state (same as spectral3.py initialization)
        self.audio_enhancer.noise_profile = np.zeros((self.audio_enhancer.frame_length // 2 + 1, 1))
        self.audio_enhancer.noise_frames_count = 0
        self.audio_enhancer.speech_frames_count = 0
        
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
            
            # Wait for recording thread to finish
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
    
    def enhance_recorded_audio(self):
        """Apply audio enhancement to the recorded file using spectral3.py method"""
        try:
            if not os.path.exists(self.temp_wav_file):
                return False
                
            # Load the recorded audio
            audio_data, sr = librosa.load(self.temp_wav_file, sr=16000)
            
            if len(audio_data) == 0:
                return False
            
            # Apply enhancement if enabled
            if self.enhancement_enabled.get():
                self.update_status("Enhancing audio quality...")
                enhanced_audio = self.audio_enhancer.enhance_full_audio(audio_data)
                
                # Save enhanced audio (same normalization as spectral3.py)
                sf.write(self.enhanced_wav_file, enhanced_audio, sr)
                
                # Update quality info
                speech_ratio = (self.audio_enhancer.speech_frames_count / 
                              max(1, self.audio_enhancer.speech_frames_count + self.audio_enhancer.noise_frames_count))
                quality_info = f"Enhancement applied: {speech_ratio:.1%} speech detected, noise reduced"
                self.update_quality_info(quality_info)
                
                return True
            else:
                # Copy original file if enhancement is disabled
                import shutil
                shutil.copy2(self.temp_wav_file, self.enhanced_wav_file)
                self.update_quality_info("Enhancement disabled - using original audio")
                return True
                
        except Exception as e:
            print(f"Audio enhancement error: {e}")
            self.update_quality_info(f"Enhancement failed: {str(e)}")
            # Copy original file as fallback
            try:
                import shutil
                shutil.copy2(self.temp_wav_file, self.enhanced_wav_file)
                return True
            except:
                return False
    
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
            
            # Enhance audio quality
            if not self.enhance_recorded_audio():
                self.update_status("Error: Failed to process audio")
                self.root.after(0, self.progress.stop)
                return
            
            # Use enhanced audio file for transcription
            audio_file_to_use = self.enhanced_wav_file if os.path.exists(self.enhanced_wav_file) else self.temp_wav_file
            
            # Get the selected language code
            selected_language = self.language_var.get()
            target_lang_code = self.language_mapping[selected_language]["code"]
            
            # Step 1: Transcribe using enhanced audio
            self.update_status("Transcribing enhanced audio...")
            result = self.whisper_model.transcribe(audio_file_to_use)
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
                generation = self.tts_model.generate(
                    input_ids=description_input_ids,
                    attention_mask=description_attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask
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
                if self.auto_play_enabled.get():
                    self.root.after(100, self.play_audio)
            self.root.after(0, enable_play)
            
        except Exception as e:
            self.root.after(0, self.progress.stop)
            self.update_status(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
    
    def play_audio(self):
        try:
            if os.name == "nt":  # Windows
                os.system(f"start {self.output_file}")
            else:  # Linux/Mac
                os.system(f"mpg123 {self.output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not play audio: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechTranslatorApp(root)
    root.mainloop()