import sounddevice


    



import pyaudio
import numpy as np
import speech_recognition as sr
import io
import os
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
from threading import Thread
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access variables
api_key = os.getenv("OPENAI_API_KEYS")

class Recorder:
    def __init__(self, gui):
        self.gui = gui
        self.prompt_result_name = ""
        self.recognizer = sr.Recognizer()
        self.chunk = 1024
        self.sample_rate = 44100
        self.channels = 1
        self.chunk_duration = 5
        self.silence_threshold = 0.01
        self.silence_duration = 0.4
        self.audio_filename = "audio.wav"
        self.transcript_file = self.get_new_transcription_filename()
        self.paragraph = ""
        self.silence_counter = 0
        self.audio_buffer = io.BytesIO()
        self.is_recording = None
        self.transcript_file_prompt = self.get_new_transcription_result_filename()

    def launch_ai_with_prompt(self, prompt):
        try:
            openai_client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1", api_key=api_key
            )

            response = openai_client.chat.completions.create(
                model="meta/llama-3.1-405b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=False,
            )

            return response.choices[0].message.content

        except Exception as e:
            print("An error occurred: {}".format(str(e).split("\n")[0]))
            return None

    def get_file_name(self, filename):
        # Extract the base name of the file (e.g., 'dev.txt')
        base_name = os.path.basename(filename)
        # Split the base name into name and extension
        name, _ = os.path.splitext(base_name)
        self.prompt_result_name = name  # Set the prompt_result_name
        print(f"Prompt result name set to: {self.prompt_result_name}")

    def get_new_transcription_filename(self):
        base_name = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if base_name:
            self.get_file_name(base_name)  # Ensure this is called to set prompt_result_name
        print(f"Selected base name: {self.prompt_result_name}")
        return base_name

    def get_new_transcription_result_filename(self):
        if not self.prompt_result_name:
            print("Error: prompt_result_name is not set.")
            return None
        
        base_name = self.prompt_result_name
        print(f"Base name: {base_name}")
        suffix = 0
        while True:
            filename = f"{base_name}ai_result_summary-{suffix}.txt" if suffix > 0 else f"{base_name}ai_result_summary.txt"
            full_path = os.path.join("/home/riter/Projects/note-taker/ai_results", filename)
            print(f"Checking filename: {full_path}")
            if not os.path.exists(full_path):
                return full_path
            suffix += 1

    def recognize_speech_from_audio(self, audio_data):
        try:
            audio = sr.AudioData(audio_data, self.sample_rate, 2)
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    def is_silent(self, audio_data):
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(np.square(audio_array)))
        return rms < self.silence_threshold

    def write_paragraph_to_file(self):
        if self.paragraph.strip():  # Ensure paragraph is not an empty string
            try:
                with open(self.transcript_file, "a") as file:
                    file.write(self.paragraph + "\n\n")
                self.gui.update_transcription(self.paragraph)
                print("Paragraph written to file:", self.paragraph)
                # self.paragraph = ""
            except Exception as e:
                print(f"Error writing paragraph to file: {e}")

    def record_and_transcribe(self):
        print("Starting real-time audio recording and transcription...")
        p = pyaudio.PyAudio()

        try:
            stream = p.open(format=pyaudio.paInt16,
                            channels=self.channels,
                            rate=self.sample_rate,
                            input=True,
                            frames_per_buffer=self.chunk,
                            stream_callback=self.audio_callback)

            while self.is_recording is not None:
                self.gui.root.update()
        except KeyboardInterrupt:
            self.stop_recording()
        except Exception as e:
            print(f"Error during recording: {e}")
            self.stop_recording()

    def stop_recording(self):
        self.is_recording = None
        self.write_paragraph_to_file()

        prompt = f"I need assistance in fully comprehending and analyzing the following question or text: {self.paragraph} \n 1. Clarification and Explanation: Break down the text thoroughly, identifying and explaining the main points, key ideas, and any complex concepts. Simplify difficult language and provide context where necessary to ensure complete understanding. \n 2. Actionable Items and Recommendations: Identify and list all actionable items, tasks, or recommendations mentioned or implied in the text. Provide a detailed explanation of each action, including why it is important, how it should be executed, and what the expected outcomes are. \n 3. Comprehensive Summary: Develop a detailed summary that encapsulates all the critical information, key themes, and insights from the text. Ensure that the summary is both concise and inclusive, covering all relevant aspects without missing any important details. \n 4. Conclusion and Next Steps: Offer a well-rounded conclusion that synthesizes the information, drawing connections between key points. Highlight the overall significance of the text, suggest next steps or actions to be taken, and provide any additional insights or reflections that can help in making informed decisions or deepening understanding."
        prompt_result = self.launch_ai_with_prompt(prompt)

        if prompt_result:
            try:
                with open(self.transcript_file_prompt, "a") as file:
                    file.write(prompt_result + "\n\n")
            except Exception as e:
                print(f"Error writing AI result to file: {e}")

        print("Stopping recording...")
        try:
            with open(self.audio_filename, "wb") as audio_file:
                audio_file.write(self.audio_buffer.getvalue())
            print(f"Final audio saved as {self.audio_filename}")
        except Exception as e:
            print(f"Error saving audio file: {e}")

        self.clean_up_files()
        self.paragraph = ""

    def pause_recording(self):
        self.is_recording = None
        print("Recording paused.")

    def resume_recording(self):
        if self.is_recording is None:  # Resume only if previously paused
            self.is_recording = True
            print("Recording resumed.")
            self.record_and_transcribe()

    def auto_save(self):
        self.write_paragraph_to_file()
        self.gui.root.after(300000, self.auto_save)  # Auto-save every 5 minutes

    def audio_callback(self, in_data, frame_count, time_info, status):
        if status:
            print("Status:", status)

        self.audio_buffer.write(in_data)

        if self.audio_buffer.tell() >= self.sample_rate * self.channels * 2 * self.chunk_duration:
            audio_data = self.audio_buffer.getvalue()
            self.audio_buffer.seek(0)
            self.audio_buffer.truncate()

            if self.is_silent(audio_data):
                self.silence_counter += self.chunk_duration
                if self.silence_counter >= self.silence_duration:
                    self.write_paragraph_to_file()
                    self.silence_counter = 0
                return
            else:
                self.silence_counter = 0

            transcription = self.recognize_speech_from_audio(audio_data)
            if transcription:
                self.paragraph += transcription + " "
                self.gui.update_transcription(transcription)
                print("Transcription:", transcription)
            else:
                print("Transcription: No speech recognized")

        return (None, pyaudio.paContinue)

    def clean_up_files(self):
        if os.path.exists(self.audio_filename):
            os.remove(self.audio_filename)
        print("Temporary files deleted.")

class RecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcription App")
        self.root.geometry("800x600")

        # Sidebar navigation
        self.sidebar = tk.Frame(self.root, width=200, bg="#333333")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        self.content_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.create_sidebar_buttons()

        # Variable to hold the current screen frame
        self.current_frame = None

        # Screen initializations
        self.transcriptions_folder = "./transcriptions"
        self.ai_results_folder = "./ai_results"
        os.makedirs(self.transcriptions_folder, exist_ok=True)
        os.makedirs(self.ai_results_folder, exist_ok=True)

        self.recorder = None
        self.recording_thread = None

    def create_sidebar_buttons(self):
        tk.Button(self.sidebar, text="Transcriptions", command=self.show_transcription_list).pack(pady=10)
        tk.Button(self.sidebar, text="AI Results", command=self.show_ai_results_list).pack(pady=10)
        tk.Button(self.sidebar, text="Record", command=self.show_recording_screen).pack(pady=10)

    def clear_content_frame(self):
        if self.current_frame:
            self.current_frame.destroy()

    def show_transcription_list(self):
        self.clear_content_frame()
        self.current_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
        self.current_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.current_frame, text="Transcriptions", font=("Arial", 16), bg="#f0f0f0").pack(pady=10)

        self.transcription_listbox = tk.Listbox(self.current_frame, selectmode=tk.SINGLE, font=("Arial", 12))
        self.transcription_listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.transcription_listbox.bind("<Double-1>", self.load_transcription_file)

        self.populate_transcription_list()

    def show_ai_results_list(self):
        self.clear_content_frame()
        self.current_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
        self.current_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.current_frame, text="AI Results", font=("Arial", 16), bg="#f0f0f0").pack(pady=10)

        self.ai_results_listbox = tk.Listbox(self.current_frame, selectmode=tk.SINGLE, font=("Arial", 12))
        self.ai_results_listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.ai_results_listbox.bind("<Double-1>", self.load_ai_result_file)

        self.populate_ai_results_list()

    def show_recording_screen(self):
        self.clear_content_frame()
        self.current_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
        self.current_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.current_frame, text="Recording Screen", font=("Arial", 16), bg="#f0f0f0").pack(pady=10)

        self.transcription_text = scrolledtext.ScrolledText(self.current_frame, wrap=tk.WORD)
        self.transcription_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        control_frame = tk.Frame(self.current_frame, bg="#f0f0f0")
        control_frame.pack(pady=10)

        self.start_button = tk.Button(control_frame, text="Start Recording", command=self.start_recording)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.pause_button = tk.Button(control_frame, text="Pause Recording", command=self.pause_recording)
        self.pause_button.pack(side=tk.LEFT, padx=5)

        self.resume_button = tk.Button(control_frame, text="Resume Recording", command=self.resume_recording)
        self.resume_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(control_frame, text="Stop Recording", command=self.stop_recording)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Initialize button states
        self.set_button_state(start=True)

    def set_button_state(self, start=True, pause=False, resume=False, stop=False):
        """Enable or disable buttons based on the recording state."""
        self.start_button.config(state=tk.NORMAL if start else tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL if pause else tk.DISABLED)
        self.resume_button.config(state=tk.NORMAL if resume else tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL if stop else tk.DISABLED)

    def start_recording(self):
        if not self.recorder:
            self.recorder = Recorder(self)

        self.recording_thread = Thread(target=self.recorder.record_and_transcribe)
        self.recording_thread.start()

        # Update button states
        self.set_button_state(start=False, pause=True, stop=True)

    def pause_recording(self):
        if self.recorder:
            self.recorder.pause_recording()

            # Update button states
            self.set_button_state(pause=False, resume=True, stop=True)

    def resume_recording(self):
        if self.recorder:
            self.recorder.resume_recording()

            # Update button states
            self.set_button_state(resume=False, pause=True, stop=True)

    def stop_recording(self):
        if self.recorder:
            self.recorder.stop_recording()
            self.recording_thread.join()
            self.recorder = None
            messagebox.showinfo("Recording Stopped", "Recording has been stopped and saved.")

            # Reset button states after stopping
            self.set_button_state(start=True)

    def update_transcription(self, text):
        self.transcription_text.insert(tk.END, text + "\n")
        self.transcription_text.yview(tk.END)

    def save_current_transcription(self):
        if self.transcription_text.get("1.0", tk.END).strip():
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if file_path:
                try:
                    with open(file_path, "w") as file:
                        file.write(self.transcription_text.get("1.0", tk.END))
                    print(f"Transcription saved as {file_path}")
                except Exception as e:
                    print(f"Error saving transcription: {e}")

    def populate_transcription_list(self):
        self.transcription_listbox.delete(0, tk.END)
        for filename in os.listdir(self.transcriptions_folder):
            if filename.endswith(".txt"):
                self.transcription_listbox.insert(tk.END, filename)

    def populate_ai_results_list(self):
        self.ai_results_listbox.delete(0, tk.END)
        for filename in os.listdir(self.ai_results_folder):
            if filename.endswith(".txt"):
                self.ai_results_listbox.insert(tk.END, filename)

    def load_transcription_file(self, event):
        selected_file = self.transcription_listbox.get(tk.ACTIVE)
        if selected_file:
            file_path = os.path.join(self.transcriptions_folder, selected_file)
            try:
                with open(file_path, "r") as file:
                    content = file.read()
                self.clear_content_frame()
                self.current_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
                self.current_frame.pack(fill=tk.BOTH, expand=True)
                tk.Label(self.current_frame, text=selected_file, font=("Arial", 16), bg="#f0f0f0").pack(pady=10)
                text_widget = scrolledtext.ScrolledText(self.current_frame, wrap=tk.WORD)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
                text_widget.insert(tk.END, content)
            except Exception as e:
                print(f"Error loading transcription file: {e}")

    def load_ai_result_file(self, event):
        selected_file = self.ai_results_listbox.get(tk.ACTIVE)
        if selected_file:
            file_path = os.path.join(self.ai_results_folder, selected_file)
            try:
                with open(file_path, "r") as file:
                    content = file.read()
                self.clear_content_frame()
                self.current_frame = tk.Frame(self.content_frame, bg="#f0f0f0")
                self.current_frame.pack(fill=tk.BOTH, expand=True)
                tk.Label(self.current_frame, text=selected_file, font=("Arial", 16), bg="#f0f0f0").pack(pady=10)
                text_widget = scrolledtext.ScrolledText(self.current_frame, wrap=tk.WORD)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
                text_widget.insert(tk.END, content)
            except Exception as e:
                print(f"Error loading AI result file: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderApp(root)
    root.mainloop()





















































# import sounddevice as sd
# import numpy as np
# import vosk
# import json
# import io
# import sys

# # Initialize Vosk model and recognizer
# model = vosk.Model("/home/riter/Projects/note-taker/vosk-model-small-en-us-0.15 ")  # Replace with the path to your Vosk model
# recognizer = vosk.KaldiRecognizer(model, 44100)

# # Audio configuration
# fs = 44100  # Sample rate
# channels = 1  # Mono audio
# chunk_duration = 5  # Duration for each chunk in seconds
# silence_threshold = 0.01  # Threshold for considering audio as silent (adjust as needed)

# def recognize_speech_from_audio(audio_data):
#     """Transcribe the audio data using Vosk."""
#     if recognizer.AcceptWaveform(audio_data):
#         result = recognizer.Result()
#         text = json.loads(result).get('text', '')
#         return text
#     else:
#         partial_result = recognizer.PartialResult()
#         text = json.loads(partial_result).get('partial', '')
#         return text

# def is_silent(audio_data):
#     """Check if the audio data is silent."""
#     audio_array = np.frombuffer(audio_data, dtype=np.int16)
#     rms = np.sqrt(np.mean(np.square(audio_array)))
#     return rms < silence_threshold

# def record_and_transcribe():
#     print("Starting real-time audio recording and transcription...")

#     # Buffer to accumulate audio data
#     audio_buffer = io.BytesIO()

#     def audio_callback(indata, frames, time_info, status):
#         """Callback function for audio stream."""
#         if status:
#             print("Status:", status)

#         # Write audio data to buffer
#         audio_buffer.write(indata.tobytes())

#         # Check if buffer has enough data for a chunk
#         if audio_buffer.tell() >= fs * channels * 2 * chunk_duration:
#             audio_data = audio_buffer.getvalue()
#             audio_buffer.seek(0)  # Reset buffer position
#             audio_buffer.truncate()  # Clear buffer
            
#             # Skip silent chunks
#             if is_silent(audio_data):
#                 print("Transcription: Silence detected, skipping chunk.")
#                 return
            
#             # Transcribe the audio chunk
#             transcription = recognize_speech_from_audio(audio_data)
#             if transcription:
#                 print("Transcription:", transcription)
#             else:
#                 print("Transcription: No speech recognized")

#     with sd.InputStream(samplerate=fs, channels=channels, dtype='int16', callback=audio_callback):
#         print("Press Ctrl+C to stop.")
#         try:
#             while True:
#                 pass  # Keep the script running to continue recording and processing
#         except KeyboardInterrupt:
#             print("Stopping recording...")
#             sys.exit()

# if __name__ == "__main__":
#     record_and_transcribe()













# # record_and_merge.py

# import pyautogui
# import cv2
# import numpy as np
# import pyaudio
# import wave
# import subprocess
# import threading
# import time
# import sounddevice as sd

# # Configurations
# video_filename = "Recording.avi"
# audio_filename = "audio.wav"
# output_filename = "output.mp4"
# duration = 60  # Duration for recording in seconds
# fps = 30.0
# resolution = (1920, 1080)
# chunk = 1024
# rate = 44100

# # Video recording function
# def record_video():
#     codec = cv2.VideoWriter_fourcc(*"XVID")
#     out = cv2.VideoWriter(video_filename, codec, fps, resolution)
#     cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Live", 480, 270)
    
#     print("Starting video recording...")
#     start_time = time.time()
#     while time.time() - start_time < duration:
#         img = pyautogui.screenshot()
#         frame = np.array(img)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         out.write(frame)
#         cv2.imshow('Live', frame)
#         if cv2.waitKey(1) == ord('q'):
#             break
    
#     out.release()
#     cv2.destroyAllWindows()
#     print("Video recording finished.")

# # Audio recording function
# def record_audio():
#     audio = pyaudio.PyAudio()
#     stream = audio.open(format=pyaudio.paInt16, channels=2, rate=rate, input=True, frames_per_buffer=chunk)
#     print("Starting audio recording...")
#     frames = []

#     start_time = time.time()
#     while time.time() - start_time < duration:
#         data = stream.read(chunk)
#         frames.append(data)
    
#     stream.stop_stream()
#     stream.close()
#     audio.terminate()
    
#     with wave.open(audio_filename, 'wb') as wf:
#         wf.setnchannels(2)
#         wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
#         wf.setframerate(rate)
#         wf.writeframes(b''.join(frames))
    
#     print("Audio recording finished.")

# # Merge video and audio function
# def merge_audio_video():
#     command = [
#         'ffmpeg',
#         '-i', video_filename,
#         '-i', audio_filename,
#         '-c:v', 'copy',
#         '-c:a', 'aac',
#         '-strict', 'experimental',
#         output_filename
#     ]
#     subprocess.run(command, check=True)
#     print(f"Files merged into {output_filename}")

# # Run recording and merging
# if __name__ == "__main__":
#     video_thread = threading.Thread(target=record_video)
#     audio_thread = threading.Thread(target=record_audio)

#     video_thread.start()
#     audio_thread.start()

#     video_thread.join()
#     audio_thread.join()

#     merge_audio_video()














# import os
# import subprocess
# import time
# from pydub import AudioSegment
# from pydub.playback import play

# def record_screen_and_audio(output_video_path, output_audio_path, duration=60):
#     # Start screen recording with audio using ffmpeg
#     command = [
#         'ffmpeg',
#         '-y',
#         '-f', 'x11grab',
#         '-r', '30',
#         '-s', '1920x1080',
#         '-i', ':0.0',  # Adjust display if needed
#         '-f', 'pulse',
#         '-i', 'default',  # Default audio device, adjust if needed
#         '-c:v', 'libx264',
#         '-c:a', 'aac',
#         '-strict', 'experimental',
#         '-t', str(duration),
#         output_video_path
#     ]
#     print("Starting screen and audio recording...")
#     process = subprocess.Popen(command)
#     time.sleep(duration)
#     process.terminate()
#     print("Recording finished.")

#     # Extract audio from the recorded video
#     extract_audio_command = [
#         'ffmpeg',
#         '-i', output_video_path,
#         '-q:a', '0',
#         '-map', 'a',
#         output_audio_path
#     ]
#     subprocess.run(extract_audio_command)
#     print("Audio extracted to", output_audio_path)


# def process_audio_for_transcription(audio_path):
#     # Convert audio to WAV format if it's not already
#     audio = AudioSegment.from_file(audio_path)
#     wav_path = "audio_for_transcription.wav"
#     audio.export(wav_path, format="wav")
#     return wav_path

# if __name__ == "__main__":
#     video_path = "screen_recording.mp4"
#     audio_path = "recorded_audio.mp3"  # Change the extension as needed
#     record_screen_and_audio(video_path, audio_path, duration=60)  # Record for 60 seconds

#     # Convert and process the audio file
#     wav_path = process_audio_for_transcription(audio_path)

#     # Now use the transcription code you provided
#     import numpy as np
#     import speech_recognition as sr
#     import io
#     import sys

#     # Initialize recognizer class
#     recognizer = sr.Recognizer()

#     # Audio configuration
#     fs = 44100  # Sample rate
#     channels = 1  # Mono audio
#     chunk_duration = 5  # Duration for each chunk in seconds
#     silence_threshold = 0.01  # Threshold for considering audio as silent (adjust as needed)
#     silence_duration = 0.4  # Seconds of silence to consider as a pause
#     transcript_file = "transcription.txt"  # Output file

#     def recognize_speech_from_audio(audio_data):
#         try:
#             audio = sr.AudioData(audio_data, fs, 2)  # Convert raw audio data to AudioData
#             text = recognizer.recognize_google(audio)
#             return text
#         except sr.UnknownValueError:
#             return None  # No speech recognized
#         except sr.RequestError:
#             return "Could not request results from Google Speech Recognition service"
#         except Exception as e:
#             return f"An unexpected error occurred: {e}"

#     def is_silent(audio_data):
#         """Check if the audio data is silent."""
#         audio_array = np.frombuffer(audio_data, dtype=np.int16)
#         rms = np.sqrt(np.mean(np.square(audio_array)))
#         return rms < silence_threshold

#     def write_paragraph_to_file(paragraph):
#         """Write the paragraph to the file."""
#         with open(transcript_file, "a") as file:
#             file.write(paragraph + "\n\n")
#         print("Paragraph written to file:", paragraph)

#     def record_and_transcribe():
#         print("Starting real-time audio recording and transcription...")

#         # Buffer to accumulate audio data
#         audio_buffer = io.BytesIO()
#         paragraph = ""
#         silence_counter = 0  # Counter to track silence duration

#         def audio_callback(indata, frames, time_info, status):
#             nonlocal paragraph, silence_counter

#             if status:
#                 print("Status:", status)

#             # Write audio data to buffer
#             audio_buffer.write(indata.tobytes())

#             # Check if buffer has enough data for a chunk
#             if audio_buffer.tell() >= fs * channels * 2 * chunk_duration:
#                 audio_data = audio_buffer.getvalue()
#                 audio_buffer.seek(0)  # Reset buffer position
#                 audio_buffer.truncate()  # Clear buffer
                
#                 # Check for silence
#                 if is_silent(audio_data):
#                     silence_counter += chunk_duration
#                     if silence_counter >= silence_duration:
#                         # End of a paragraph due to prolonged silence
#                         if paragraph.strip():
#                             write_paragraph_to_file(paragraph)
#                             paragraph = ""  # Reset for new paragraph
#                         silence_counter = 0  # Reset silence counter
#                     return
#                 else:
#                     silence_counter = 0  # Reset silence counter if speech is detected
                
#                 # Transcribe the audio chunk
#                 transcription = recognize_speech_from_audio(audio_data)
#                 if transcription:
#                     paragraph += transcription + " "
#                     print("Transcription:", transcription)
#                 else:
#                     print("Transcription: No speech recognized")

#         with sd.InputStream(samplerate=fs, channels=channels, dtype='int16', callback=audio_callback):
#             print("Press Ctrl+C to stop.")
#             try:
#                 while True:
#                     pass  # Keep the script running to continue recording and processing
#             except KeyboardInterrupt:
#                 # Final write if there's any leftover text
#                 if paragraph.strip():
#                     write_paragraph_to_file(paragraph)
#                 print("Stopping recording...")
#                 sys.exit()

#     if __name__ == "__main__":
#         # Process the audio file for transcription
#         record_and_transcribe()









# # import required modules 
# from os import path 
# from pydub import AudioSegment 
  
# # assign files 
# input_file = "/home/riter/Projects/note-taker/i-want-to-work-2.mp3"
# output_file = "/home/riter/Projects/note-taker/i-want-to-work-2.wav"
  
# # convert mp3 file to wav file 
# sound = AudioSegment.from_mp3(input_file) 
# sound.export(output_file, format="wav") 