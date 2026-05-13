import argparse
from faster_whisper import WhisperModel
from datetime import timedelta
import os

#config
task = "transcribe" 
#tasks: translate, transcribe

def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milliseconds = (seconds - int(seconds)) * 1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"

def transcribe_video(input_file, lang):
    model_size = "medium"  # other models: small, medium, large-v2, tiny

    # Run on CPU with INT8
    #model = WhisperModel(model_size, device="cpu", cpu_threads=12, compute_type="int8")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    # Remove task="translate" if you want the original language
    segments, info = model.transcribe(input_file, 
    language=lang,
    no_speech_threshold=None,   # 提高對「非說話」的容忍度
    log_prob_threshold=None,   # 禁用信心值門檻
    compression_ratio_threshold=None, # 禁用壓縮比門檻（防止長重複內容被刪除）
    word_timestamps=True,
    beam_size=5, 
    task=task, 
    vad_filter=False)
    print("Detected language '{}' with probability {:.2f}".format(info.language, info.language_probability), "Duration:",info.duration)

    srt_filename = os.path.splitext(input_file)[0] + '.'+info.language+'.srt'

    with open(srt_filename, 'w', encoding='utf-8') as srt_file:
        for segment in segments:
            start_time = format_time(segment.start)
            end_time = format_time(segment.end)
            text = segment.text
            segment_id = segment.id + 1
            line_out = f"{segment_id}\n{start_time} --> {end_time}\n{text.lstrip()}\n\n"
            print(line_out)
            srt_file.write(f"{segment_id}\n{start_time} --> {end_time}\n{text.lstrip()}\n\n")
            srt_file.flush()  # i flush the file buffer so i don't lose data if it crashes midway

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio from a video file and generate an SRT file.")
    parser.add_argument("-lang", default="en", required = False, help="Video language")
    parser.add_argument("input_file", help="Path to the video file for transcription")
    args = parser.parse_args()
    print(args)
    transcribe_video(args.input_file, args.lang)

if __name__ == "__main__":
    main()
