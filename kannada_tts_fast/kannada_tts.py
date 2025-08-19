#!/usr/bin/env python3
"""
Surgical Kannada TTS - More targeted CUDA patches
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path

# Set environment variables but don't break torch.device
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['FORCE_CPU'] = '1'

# Import torch
import torch

print("🔧 Applying surgical CUDA patches...")

# Store original functions before patching
original_cuda_is_available = torch.cuda.is_available
original_cuda_device_count = torch.cuda.device_count

# Only patch specific CUDA functions, leave torch.device alone
def fake_cuda_available():
    return False

def fake_cuda_device_count():
    return 0

# Apply minimal patches
torch.cuda.is_available = fake_cuda_available
torch.cuda.device_count = fake_cuda_device_count

# Patch tensor .cuda() and .to() methods
original_tensor_cuda = torch.Tensor.cuda
original_tensor_to = torch.Tensor.to

def patched_tensor_cuda(self, device=None, non_blocking=False):
    # Always return CPU version
    return self.cpu()

def patched_tensor_to(self, *args, **kwargs):
    # Intercept cuda device requests
    if args:
        device = args[0]
        if isinstance(device, str) and 'cuda' in device.lower():
            return original_tensor_to(self, 'cpu', *args[1:], **kwargs)
        elif hasattr(device, 'type') and 'cuda' in str(device.type):
            return original_tensor_to(self, 'cpu', *args[1:], **kwargs)
    
    if 'device' in kwargs:
        device = kwargs['device']
        if isinstance(device, str) and 'cuda' in device.lower():
            kwargs['device'] = 'cpu'
        elif hasattr(device, 'type') and 'cuda' in str(device.type):
            kwargs['device'] = 'cpu'
    
    return original_tensor_to(self, *args, **kwargs)

# Apply tensor patches
torch.Tensor.cuda = patched_tensor_cuda
torch.Tensor.to = patched_tensor_to

print("✅ Surgical patches applied - torch.device preserved")

class SurgicalKannadaTTS:
    def __init__(self):
        self.setup_paths()
        if not self.prepare_models():
            print("⚠️  Model preparation had issues, but will try synthesis anyway...")
    
    def setup_paths(self):
        """Setup directory paths"""
        self.base_dir = Path.cwd()
        self.kn_dir = self.base_dir / 'kn'
        self.cpu_dir = self.base_dir / 'kn_cpu_patched'
        
        print(f"📁 Working directory: {self.base_dir}")
        print(f"📁 CPU models will be in: {self.cpu_dir}")
    
    def check_files(self):
        """Check if original files exist"""
        required = [
            self.kn_dir / 'fastpitch' / 'best_model.pth',
            self.kn_dir / 'fastpitch' / 'config.json',
            self.kn_dir / 'hifigan' / 'best_model.pth',
            self.kn_dir / 'hifigan' / 'config.json',
            self.kn_dir / 'fastpitch' / 'speakers.pth'
        ]
        
        missing = [f for f in required if not f.exists()]
        if missing:
            print("❌ Missing files:")
            for f in missing:
                print(f"   {f}")
            return False
        
        print("✅ All required files found")
        return True
    
    def convert_tensor_to_cpu(self, obj):
        """Recursively convert tensors to CPU"""
        if torch.is_tensor(obj):
            return obj.cpu().detach()
        elif isinstance(obj, dict):
            return {key: self.convert_tensor_to_cpu(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self.convert_tensor_to_cpu(item) for item in obj)
        else:
            return obj
    
    def convert_model_cpu(self, src_path, dst_path):
        """Convert model to guaranteed CPU version"""
        try:
            print(f"   🔄 Converting {src_path.name}...")
            
            # Use string mapping to avoid torch.device issues
            checkpoint = torch.load(src_path, map_location='cpu', weights_only=False)
            
            # Recursively convert all tensors to CPU
            checkpoint_cpu = self.convert_tensor_to_cpu(checkpoint)
            
            # Ensure parent directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save CPU version
            torch.save(checkpoint_cpu, dst_path)
            print(f"   ✅ {src_path.name} converted to CPU")
            return True
            
        except Exception as e:
            print(f"   ❌ Error converting {src_path.name}: {e}")
            # Try direct copy as fallback
            try:
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"   ✅ {src_path.name} copied directly (fallback)")
                return True
            except Exception as e2:
                print(f"   ❌ Direct copy also failed: {e2}")
                return False
    
    def patch_config(self, src_path, dst_path):
        """Create patched config that forces CPU"""
        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Force all CPU settings
            config['use_cuda'] = False
            config['gpu'] = False
            
            if 'model_args' in config:
                config['model_args']['use_cuda'] = False
                config['model_args']['gpu'] = False
            
            # Remove any CUDA references
            for key in list(config.keys()):
                if 'cuda' in key.lower() or 'gpu' in key.lower():
                    if key not in ['use_cuda']:
                        config.pop(key, None)
            
            # Ensure parent directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(dst_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ {src_path.name} patched for CPU")
            return True
            
        except Exception as e:
            print(f"   ❌ Error patching {src_path.name}: {e}")
            return False
    
    def prepare_models(self):
        """Prepare CPU-only models"""
        print("\n🔄 Preparing CPU-patched models...")
        
        if not self.check_files():
            return False
        
        # Create directories
        (self.cpu_dir / 'fastpitch').mkdir(parents=True, exist_ok=True)
        (self.cpu_dir / 'hifigan').mkdir(parents=True, exist_ok=True)
        
        # First patch configs
        configs = [
            ('fastpitch/config.json', 'FastPitch config'),
            ('hifigan/config.json', 'HiFi-GAN config')
        ]
        
        for config_path, desc in configs:
            src = self.kn_dir / config_path
            dst = self.cpu_dir / config_path
            
            if not self.patch_config(src, dst):
                return False
        
        # Then convert models
        models = [
            ('fastpitch/best_model.pth', 'FastPitch model'),
            ('hifigan/best_model.pth', 'HiFi-GAN model'),
            ('fastpitch/speakers.pth', 'Speakers file')
        ]
        
        success = True
        for model_path, desc in models:
            src = self.kn_dir / model_path
            dst = self.cpu_dir / model_path
            
            if not self.convert_model_cpu(src, dst):
                success = False
        
        if success:
            print("   🎉 All models prepared successfully!")
        else:
            print("   ⚠️  Some model preparation failed, but will try synthesis...")
        
        return success
    
    def get_available_speakers(self, synthesizer):
        """Get available speakers from the model"""
        try:
            if hasattr(synthesizer.tts_model, 'speaker_manager') and synthesizer.tts_model.speaker_manager:
                speakers = synthesizer.tts_model.speaker_manager.speaker_names
                if speakers:
                    print(f"📢 Available speakers: {speakers}")
                    return speakers
                else:
                    print("📢 Speaker manager exists but no speaker names found")
            
            # Try to get speaker count from config
            if hasattr(synthesizer, 'tts_config') and hasattr(synthesizer.tts_config, 'num_speakers'):
                num_speakers = synthesizer.tts_config.num_speakers
                if num_speakers and num_speakers > 1:
                    print(f"📢 Model has {num_speakers} speakers (using index 0)")
                    return list(range(num_speakers))
            
            print("📢 No speaker information found, will try default")
            return None
            
        except Exception as e:
            print(f"📢 Error getting speakers: {e}")
            return None

    def synthesize_with_original_models(self, text, output_path='output.wav'):
        """Try synthesis with original models"""
        print(f"\n🎵 Attempting synthesis with original models...")
        print(f"📝 Text: {text}")
        print(f"💾 Output: {output_path}")
        
        try:
            # Import TTS after patches are applied
            from TTS.utils.synthesizer import Synthesizer
            
            print("⏳ Initializing synthesizer with original models...")
            
            # Use original models with CPU forcing
            synthesizer = Synthesizer(
                tts_checkpoint=str(self.kn_dir / 'fastpitch' / 'best_model.pth'),
                tts_config_path=str(self.kn_dir / 'fastpitch' / 'config.json'),
                vocoder_checkpoint=str(self.kn_dir / 'hifigan' / 'best_model.pth'),
                vocoder_config=str(self.kn_dir / 'hifigan' / 'config.json'),
                use_cuda=False
            )
            
            # Check for available speakers
            speakers = self.get_available_speakers(synthesizer)
            
            print("⏳ Synthesizing audio...")
            start_time = time.time()
            
            # Generate audio with speaker handling
            wav = None
            if speakers:
                # Try with first speaker
                speaker = speakers[0] if isinstance(speakers, list) else speakers[0]
                print(f"🎤 Using speaker: {speaker}")
                try:
                    wav = synthesizer.tts(text, speaker_name=speaker)
                except:
                    try:
                        wav = synthesizer.tts(text, speaker_idx=0)
                    except:
                        wav = synthesizer.tts(text)
            else:
                # Try different approaches for multi-speaker model
                try:
                    wav = synthesizer.tts(text, speaker_idx=0)
                    print("🎤 Used speaker_idx=0")
                except:
                    try:
                        wav = synthesizer.tts(text, speaker_name="default")
                        print("🎤 Used speaker_name='default'")
                    except:
                        wav = synthesizer.tts(text)
                        print("🎤 Used default synthesis")
            
            # Save audio
            synthesizer.save_wav(wav, output_path)
            
            synthesis_time = time.time() - start_time
            
            # Check output
            output_file = Path(output_path)
            if output_file.exists() and output_file.stat().st_size > 1000:
                file_size = output_file.stat().st_size
                
                # Calculate duration safely
                try:
                    # Try different ways to get sample rate
                    sample_rate = None
                    if hasattr(synthesizer, 'ap') and hasattr(synthesizer.ap, 'sample_rate'):
                        sample_rate = synthesizer.ap.sample_rate
                    elif hasattr(synthesizer, 'tts_config') and hasattr(synthesizer.tts_config, 'audio'):
                        sample_rate = synthesizer.tts_config.audio.get('sample_rate', 22050)
                    elif hasattr(synthesizer, 'vocoder_ap') and hasattr(synthesizer.vocoder_ap, 'sample_rate'):
                        sample_rate = synthesizer.vocoder_ap.sample_rate
                    else:
                        sample_rate = 22050  # Default
                    
                    duration = len(wav) / sample_rate
                    rtf_text = f"📈 Real-time factor: {synthesis_time/duration:.1f}x"
                except:
                    duration = "unknown"
                    rtf_text = ""
                
                print(f"\n🎉 SUCCESS!")
                print(f"✅ Audio file: {output_file.absolute()}")
                print(f"📊 File size: {file_size:,} bytes")
                print(f"⏱️  Synthesis time: {synthesis_time:.1f} seconds")
                if duration != "unknown":
                    print(f"🎵 Audio duration: {duration:.1f} seconds")
                    print(rtf_text)
                print(f"🔊 Play with: open '{output_path}'")
                
                return True
            else:
                print("❌ Output file was not created or is empty")
                return False
                
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def synthesize_with_converted_models(self, text, output_path='output.wav'):
        """Try synthesis with converted models"""
        print(f"\n🎵 Attempting synthesis with converted models...")
        print(f"📝 Text: {text}")
        print(f"💾 Output: {output_path}")
        
        # Check if converted models exist
        required_files = [
            self.cpu_dir / 'fastpitch' / 'best_model.pth',
            self.cpu_dir / 'fastpitch' / 'config.json',
            self.cpu_dir / 'hifigan' / 'best_model.pth',
            self.cpu_dir / 'hifigan' / 'config.json'
        ]
        
        if not all(f.exists() for f in required_files):
            print("❌ Converted models not available")
            return False
        
        try:
            from TTS.utils.synthesizer import Synthesizer
            
            print("⏳ Initializing synthesizer with converted models...")
            
            synthesizer = Synthesizer(
                tts_checkpoint=str(self.cpu_dir / 'fastpitch' / 'best_model.pth'),
                tts_config_path=str(self.cpu_dir / 'fastpitch' / 'config.json'),
                vocoder_checkpoint=str(self.cpu_dir / 'hifigan' / 'best_model.pth'),
                vocoder_config=str(self.cpu_dir / 'hifigan' / 'config.json'),
                use_cuda=False
            )
            
            # Check for available speakers
            speakers = self.get_available_speakers(synthesizer)
            
            print("⏳ Synthesizing audio...")
            start_time = time.time()
            
            # Generate audio with speaker handling
            wav = None
            if speakers:
                # Try with first speaker
                speaker = speakers[0] if isinstance(speakers, list) else speakers[0]
                print(f"🎤 Using speaker: {speaker}")
                try:
                    wav = synthesizer.tts(text, speaker_name=speaker)
                except:
                    try:
                        wav = synthesizer.tts(text, speaker_idx=0)
                    except:
                        wav = synthesizer.tts(text)
            else:
                # Try different approaches for multi-speaker model
                try:
                    wav = synthesizer.tts(text, speaker_idx=0)
                    print("🎤 Used speaker_idx=0")
                except:
                    try:
                        wav = synthesizer.tts(text, speaker_name="default")
                        print("🎤 Used speaker_name='default'")
                    except:
                        wav = synthesizer.tts(text)
                        print("🎤 Used default synthesis")
            
            synthesizer.save_wav(wav, output_path)
            
            synthesis_time = time.time() - start_time
            
            output_file = Path(output_path)
            if output_file.exists() and output_file.stat().st_size > 1000:
                file_size = output_file.stat().st_size
                
                # Calculate duration safely
                try:
                    # Try different ways to get sample rate
                    sample_rate = None
                    if hasattr(synthesizer, 'ap') and hasattr(synthesizer.ap, 'sample_rate'):
                        sample_rate = synthesizer.ap.sample_rate
                    elif hasattr(synthesizer, 'tts_config') and hasattr(synthesizer.tts_config, 'audio'):
                        sample_rate = synthesizer.tts_config.audio.get('sample_rate', 22050)
                    elif hasattr(synthesizer, 'vocoder_ap') and hasattr(synthesizer.vocoder_ap, 'sample_rate'):
                        sample_rate = synthesizer.vocoder_ap.sample_rate
                    else:
                        sample_rate = 22050  # Default
                    
                    duration = len(wav) / sample_rate
                    rtf_text = f"📈 Real-time factor: {synthesis_time/duration:.1f}x"
                except:
                    duration = "unknown"
                    rtf_text = ""
                
                print(f"\n🎉 SUCCESS with converted models!")
                print(f"✅ Audio file: {output_file.absolute()}")
                print(f"📊 File size: {file_size:,} bytes")
                print(f"⏱️  Synthesis time: {synthesis_time:.1f} seconds")
                if duration != "unknown":
                    print(f"🎵 Audio duration: {duration:.1f} seconds")
                    print(rtf_text)
                print(f"🔊 Play with: open '{output_path}'")
                
                return True
            else:
                print("❌ Output file was not created or is empty")
                return False
                
        except Exception as e:
            print(f"❌ Converted model synthesis failed: {e}")
            return False
    
    def synthesize(self, text, output_path='output.wav'):
        """Main synthesis method"""
        # Try original models first (most likely to work)
        if self.synthesize_with_original_models(text, output_path):
            return True
        
        print("\n🔄 Original models failed, trying converted models...")
        if self.synthesize_with_converted_models(text, output_path):
            return True
        
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("🎙️ Surgical Kannada TTS")
        print("=" * 25)
        print("Usage:")
        print("  python3 surgical_tts.py 'ನಿಮ್ಮ ಪಠ್ಯ' [output.wav]")
        print()
        print("Examples:")
        print("  python3 surgical_tts.py 'ನಮಸ್ಕಾರ!'")
        print("  python3 surgical_tts.py 'ಬೆಂಗಳೂರು ನಗರ' city.wav")
        return
    
    text = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'surgical_output.wav'
    
    print("🎙️ Surgical Kannada TTS")
    print("=" * 25)
    print(f"📝 Text: {text}")
    print(f"💾 Output: {output_file}")
    
    # Check PyTorch
    print(f"🐍 PyTorch: {torch.__version__}")
    print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    
    # Initialize and run
    tts = SurgicalKannadaTTS()
    
    if tts.synthesize(text, output_file):
        print(f"\n🎉 Success! Your Kannada audio is ready!")
        print(f"🔊 To play: open '{output_file}'")
    else:
        print(f"\n❌ All synthesis methods failed")

if __name__ == "__main__":
    main()