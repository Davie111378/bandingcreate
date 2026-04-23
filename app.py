from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, session
import os
import uuid
import cv2
import numpy as np
import zipfile
import io

app = Flask(__name__)
app.secret_key = 'banding_lab_secret_key'

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def apply_banding_effect(frame, params, mode='single', frame_idx=0, fps=24.0):
    height, width = frame.shape[:2]
    
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    img_float = frame.astype(np.float32) / 255.0
    
    if mode == 'single':
        light_freq = params.get('frequency', 100.0)
        modulation_depth = params.get('modulation_depth', 0.85)
        line_time_us = params.get('line_time', 100.0)
        exposure_ms = params.get('exposure_time', 8.0)
        
        x = light_freq * exposure_ms * 1e-3
        sinc_factor = np.sinc(x)
        phase_per_frame = 2 * np.pi * light_freq / fps
        
        rows = np.arange(height)
        t_start = rows * (line_time_us * 1e-6)
        amplitude = modulation_depth * sinc_factor
        current_phase = frame_idx * phase_per_frame
        phase = 2 * np.pi * light_freq * t_start + current_phase
        
        gains_1d = 1.0 + amplitude * np.sin(phase)
        gains = gains_1d.reshape(-1, 1, 1)
        
        banded = np.clip(img_float * gains, 0.0, 1.0)
        result = (banded * 255).astype(np.uint8)
    
    else:
        led_freq = params.get('led_frequency', 115.0)
        led_mod_depth = params.get('led_modulation_depth', 0.40)
        led_line_time_us = params.get('led_line_time', 100.0)
        led_exposure_ms = params.get('led_exposure_time', 5.0)
        led_weight = params.get('led_weight', 0.6)
        
        fl_freq = params.get('fl_frequency', 100.0)
        fl_mod_depth = params.get('fl_modulation_depth', 0.50)
        fl_line_time_us = params.get('fl_line_time', 80.0)
        fl_exposure_ms = params.get('fl_exposure_time', 5.0)
        fl_weight = params.get('fl_weight', 0.4)
        
        x_led = led_freq * led_exposure_ms * 1e-3
        alpha_led = np.sinc(x_led)
        x_fl = fl_freq * fl_exposure_ms * 1e-3
        alpha_fl = np.sinc(x_fl)
        
        tau_led = led_line_time_us * 1e-6
        t_frame = frame_idx / fps
        phi_led = 2 * np.pi * led_freq * t_frame
        
        rows = np.arange(height).reshape(-1, 1)
        cols = np.arange(width).reshape(1, -1)
        pixel_time_led = tau_led / width
        t_row_led = rows * tau_led
        t_col_led = cols * pixel_time_led * height
        time_grid_led = t_row_led + t_col_led
        phase_led = 2 * np.pi * led_freq * time_grid_led + phi_led
        amp_led = led_mod_depth * alpha_led
        G_led = 1.0 + amp_led * np.sin(phase_led)
        
        tau_fl = fl_line_time_us * 1e-6
        phi_fl = 2 * np.pi * fl_freq * t_frame
        
        rows_fl = np.arange(height)
        t_start_fl = rows_fl * tau_fl
        phase_fl = 2 * np.pi * fl_freq * t_start_fl + phi_fl
        amp_fl = fl_mod_depth * alpha_fl
        G_fl_1d = 1.0 + amp_fl * np.sin(phase_fl)
        G_fl = G_fl_1d.reshape(-1, 1).repeat(width, axis=1)
        
        G_mix = (led_weight * G_led + fl_weight * G_fl) / (led_weight + fl_weight)
        
        if len(img_float.shape) == 3:
            G_mix = G_mix[:, :, np.newaxis]
        
        result = np.clip(img_float * G_mix, 0.0, 1.0)
        result = (result * 255).astype(np.uint8)
    
    return result

def generate_frame_name(base_name, frame_idx, total_frames, params, mode):
    timestamp = f"{frame_idx / total_frames:.2f}s"
    if mode == 'single':
        freq = params.get('frequency', 115.0)
        depth = params.get('modulation_depth', 0.85)
        return f"{base_name}_frame{frame_idx:02d}_t{timestamp}_D{int(depth*100):02d}_F{int(freq)}Hz.png"
    else:
        led_freq = params.get('led_frequency', 115.0)
        led_depth = params.get('led_modulation_depth', 0.85)
        fl_freq = params.get('fl_frequency', 100.0)
        fl_depth = params.get('fl_modulation_depth', 0.5)
        return f"{base_name}_frame{frame_idx:02d}_t{timestamp}_LED{int(led_freq)}Hz_{int(led_depth*100)}_FL{int(fl_freq)}Hz_{int(fl_depth*100)}.png"

@app.route('/')
def index():
    session.clear()
    return render_template('step1.html')

@app.route('/step1', methods=['GET', 'POST'])
def step1():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return jsonify({'error': '请选择视频文件'}), 400
        
        file = request.files['video_file']
        if file.filename == '':
            return jsonify({'error': '请选择视频文件'}), 400
        
        if file and allowed_file(file.filename):
            task_id = str(uuid.uuid4())
            filename = f"{task_id}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            cap = cv2.VideoCapture(filepath)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            session['task_id'] = task_id
            session['video_path'] = filepath
            session['video_name'] = file.filename.rsplit('.', 1)[0]
            session['total_frames'] = total_frames
            session['fps'] = fps
            session['duration'] = duration
            session['width'] = width
            session['height'] = height
            session['extract_frames'] = int(request.form.get('extract_frames', 15))
            
            return jsonify({'success': True, 'redirect': url_for('step2')})
    
    return render_template('step1.html')

@app.route('/step2', methods=['GET', 'POST'])
def step2():
    if 'task_id' not in session:
        return redirect(url_for('step1'))
    
    if request.method == 'POST':
        params = request.json
        session['mode'] = params.get('mode', 'single')
        session['params'] = params
        return jsonify({'success': True, 'redirect': url_for('step3')})
    
    return render_template('step2.html', 
                         video_name=session.get('video_name'),
                         total_frames=session.get('total_frames'),
                         duration=session.get('duration'),
                         extract_frames=session.get('extract_frames'))

@app.route('/step3', methods=['GET', 'POST'])
def step3():
    if 'task_id' not in session or 'params' not in session:
        return redirect(url_for('step1'))
    
    if request.method == 'POST':
        task_id = session['task_id']
        video_path = session['video_path']
        video_name = session['video_name']
        extract_frames = session['extract_frames']
        mode = session['mode']
        params = session['params']
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        step = max(1, total_frames // extract_frames)
        output_files = []
        
        for i in range(extract_frames):
            frame_idx = min(i * step, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                processed_frame = apply_banding_effect(frame, params, mode, i)
                frame_name = generate_frame_name(video_name, i, extract_frames, params, mode)
                output_path = os.path.join(OUTPUT_FOLDER, frame_name)
                cv2.imwrite(output_path, processed_frame)
                output_files.append(frame_name)
        
        cap.release()
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for f in output_files:
                zipf.write(os.path.join(OUTPUT_FOLDER, f), f)
        
        zip_buffer.seek(0)
        session['output_files'] = output_files
        
        return jsonify({'success': True, 'files': output_files})
    
    return render_template('step3.html',
                         video_name=session.get('video_name'),
                         mode=session.get('mode'),
                         extract_frames=session.get('extract_frames'))

@app.route('/download/<filename>')
def download_file(filename):
    import os
    full_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(full_path):
        return jsonify({'error': '文件不存在'}), 404
    
    from flask import make_response, send_file
    response = make_response(send_file(full_path, as_attachment=True))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/download_zip')
def download_zip():
    if 'output_files' not in session:
        return redirect(url_for('step1'))
    
    output_files = session['output_files']
    video_name = session.get('video_name', 'output')
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in output_files:
            zipf.write(os.path.join(OUTPUT_FOLDER, f), f)
    
    zip_buffer.seek(0)
    from flask import make_response
    response = make_response(zip_buffer.getvalue())
    response.headers['Content-Type'] = 'application/zip'
    response.headers['Content-Disposition'] = f'attachment; filename={video_name}_banding.zip'
    return response

@app.route('/preview', methods=['POST'])
def preview():
    if 'video_path' not in session:
        return jsonify({'error': '请先上传视频'}), 400
    
    params = request.json
    mode = params.get('mode', 'single')
    
    cap = cv2.VideoCapture(session['video_path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({'error': '无法读取视频帧'}), 400
    
    processed_frame = apply_banding_effect(frame, params, mode)
    
    _, buffer = cv2.imencode('.png', processed_frame)
    import base64
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'success': True, 'image': img_str})

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)