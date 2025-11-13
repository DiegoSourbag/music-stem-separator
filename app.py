
import os
import sys
import threading
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify, url_for, redirect, send_from_directory

from music_processor import MusicProcessor

from karaoke import KaraokeCreator

from smart_processor import SmartProcessor

from youtube_downloader import YouTubeAudioDownloader

# Set console encoding for Windows
if sys.platform == 'win32':
    import io
    # Only wrap if not already wrapped
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

app = Flask(__name__)

# --- State Management ---
# A simple in-memory store for job status.
# For a real-world app, you'd use a database or a task queue like Celery.
job_status = {
    'is_running': False,
    'progress': 0,
    'total': 0,
    'current_task': 'Waiting to start...',
    'error': None,
    'output_files': []
}

# --- Background Processing ---
def run_processing_job(urls, mode, model, high_performance):
    """The actual processing function that runs in a background thread."""
    global job_status
    job_status['is_running'] = True
    job_status['progress'] = 0
    job_status['total'] = len(urls)
    job_status['error'] = None
    job_status['output_files'] = []

    processor = None
    if mode == 'karaoke':
        processor = KaraokeCreator(model=model, high_performance=high_performance)
    elif mode == 'download':
        processor = YouTubeAudioDownloader(output_dir='downloads', format='mp3')
    else: # '4-stem' or '6-stem'
        processor = MusicProcessor(model=model, high_performance=high_performance)

    for i, url in enumerate(urls):
        job_status['current_task'] = f"Processing URL {i+1}/{len(urls)}: {url}"
        try:
            if mode == 'karaoke':
                result = processor.create_from_youtube(url=url, keep_original=False)
                job_status['output_files'].extend(result.values())
            elif mode == 'download':
                result_file = processor.download(url)
                job_status['output_files'].append(result_file)
            else:
                _, stems = processor.process_from_youtube(url=url, keep_original=False)
                job_status['output_files'].extend(stems.values())

        except Exception as e:
            print(f"Error processing {url}: {e}", file=sys.stderr)
            job_status['error'] = f"Failed on URL {i+1}: {e}"
            job_status['is_running'] = False
            return

        # Update progress after successful completion
        job_status['progress'] = i + 1

    job_status['is_running'] = False
    job_status['current_task'] = "All tasks completed!"


# --- Routes ---
@app.route('/')
def index():
    """Main page with the URL input form."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """
    Receives form data, calculates estimates, and shows a confirmation page.
    """
    urls_text = request.form.get('urls', '')
    action = request.form.get('action', '')

    urls = [url.strip() for url in urls_text.splitlines() if url.strip()]

    if not urls or not action:
        return "Please provide URLs and select an action.", 400

    model = 'htdemucs_ft' # Default for karaoke and 4-stem
    if action == '6-stem':
        model = 'htdemucs_6s'

    return render_template('confirm.html', urls=urls, action=action, model=model)


@app.route('/estimate', methods=['POST'])
def estimate():
    """
    Asynchronously estimates processing time.
    """
    urls_text = request.json.get('urls', '')
    model = request.json.get('model', 'htdemucs_ft')
    urls = [url.strip() for url in urls_text.splitlines() if url.strip()]

    if not urls:
        return jsonify({'error': 'No URLs provided'}), 400

    total_duration = 0
    smart_processor = SmartProcessor(verbose=False)
    for url in urls:
        duration = smart_processor.get_youtube_duration(url)
        if duration:
            total_duration += duration

    estimate_data = smart_processor.estimate_processing_time(total_duration, model)
    return jsonify(estimate_data)



@app.route('/run', methods=['POST'])
def run():
    """
    Starts the background processing job.
    """
    global job_status
    if job_status['is_running']:
        return "A job is already in progress.", 400

    urls_text = request.form.get('urls', '')
    action = request.form.get('action', '')
    model = request.form.get('model', 'htdemucs_ft')

    urls = [url.strip() for url in urls_text.splitlines() if url.strip()]

    # For simplicity, we'll assume high performance is false.
    # This could be a user option.
    high_performance = False

    # Start the background thread
    thread = threading.Thread(target=run_processing_job, args=(urls, action, model, high_performance))
    thread.daemon = True
    thread.start()

    return redirect(url_for('progress_page'))

@app.route('/progress')
def progress_page():
    """Displays the progress of the running job."""
    return render_template('progress.html')

@app.route('/status')
def status():
    """API endpoint to get the current job status."""
    global job_status
    return jsonify(job_status)

@app.route('/results')
def results():
    """Displays the final list of output files."""
    global job_status
    # This is a simplified results page.
    # In a real app, you'd pass job results more robustly.
    return render_template('results.html', files=job_status['output_files'])


@app.route('/outputs/<path:filepath>')
def download_file(filepath):
    """
    Serves files from the project's root output directories.
    """
    return send_from_directory('.', filepath, as_attachment=True)



if __name__ == '__main__':
    # Ensure output directories exist
    Path('downloads').mkdir(exist_ok=True)
    Path('separated').mkdir(exist_ok=True)
    Path('karaoke').mkdir(exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5001)
