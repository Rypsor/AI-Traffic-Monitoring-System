# Corrección 1: Añadir estas líneas al inicio para solucionar el error OMP
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import random
import csv

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- La clase VehicleTracker se mantiene igual, estaba correcta ---
class VehicleTracker:
    def __init__(self, max_age=7, fps=None):
        self.tracker = DeepSort(max_age=max_age)
        self.track_history = {}  # Historial de posiciones por track
        self.track_speeds = {}   # Velocidades de los tracks
        self.fps = fps  # frames por segundo del video
    
    def update(self, frame, detections):
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            
            ltrb = track.to_ltrb()
            center_x = (ltrb[0] + ltrb[2]) / 2
            center_y = (ltrb[1] + ltrb[3]) / 2
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
                self.track_speeds[track_id] = []
            
            self.track_history[track_id].append((int(center_x), int(center_y)))
            
            if len(self.track_history[track_id]) > 1:
                last_point = self.track_history[track_id][-1]
                prev_point = self.track_history[track_id][-2]
                
                pixel_distance = np.sqrt(
                    (last_point[0] - prev_point[0])**2 + 
                    (last_point[1] - prev_point[1])**2
                )
                
                if self.fps is not None and self.fps > 0:
                    speed_pixels_per_second = pixel_distance * self.fps
                    self.track_speeds[track_id].append(speed_pixels_per_second)
                else:
                    self.track_speeds[track_id].append(pixel_distance)
        
        return tracks

# --- La clase VideoProcessor tiene un pequeño ajuste para el cálculo de velocidad ---
class VideoProcessor:
    def __init__(self, video_path, max_resolution=1280):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("No se pudo abrir el video")
        
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Si el video no tiene FPS (ej. una imagen), asignar un valor por defecto
        if self.original_fps == 0:
            self.original_fps = 30.0
            
        self.new_width, self.new_height = self.calcular_nueva_resolucion(max_resolution)
        self.frame_skip = self.calcular_frame_skip()
    
    def calcular_nueva_resolucion(self, max_resolution):
        if max(self.original_width, self.original_height) <= max_resolution:
            return self.original_width, self.original_height
        
        if self.original_width > self.original_height:
            scale = max_resolution / self.original_width
        else:
            scale = max_resolution / self.original_height
        
        new_width = int(self.original_width * scale)
        new_height = int(self.original_height * scale)
        new_width = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1
        return new_width, new_height
    
    def calcular_frame_skip(self):
        if self.original_fps > 35:
            return 2
        return 1
    
    def procesar_video(self, model, output_video_path, confidence_threshold=0.5, max_age=3):
        effective_fps = self.original_fps / self.frame_skip
        vehicle_tracker = VehicleTracker(max_age=max_age, fps=effective_fps)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, effective_fps, (self.new_width, self.new_height))
        
        track_colors = {}
        def get_track_color(track_id):
            if track_id not in track_colors:
                track_colors[track_id] = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            return track_colors[track_id]
        
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue
            
            frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
            results = model(frame_resized, conf=confidence_threshold)
            
            detections_for_tracker = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                    conf = box.conf[0]
                    detections_for_tracker.append((bbox, conf, "vehicle"))
            
            tracks = vehicle_tracker.update(frame_resized, detections_for_tracker)
            frame_with_tracking = frame_resized.copy()
            
            for track_id, history in vehicle_tracker.track_history.items():
                color = get_track_color(track_id)
                if len(history) > 1:
                    for i in range(1, len(history)):
                        cv2.line(frame_with_tracking, history[i-1], history[i], color, 2)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                color = get_track_color(track_id)
                ltrb = track.to_ltrb()
                
                cv2.rectangle(frame_with_tracking, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), color, 2)
                
                # Corrección 3: Corregir el cálculo de la velocidad promedio
                if track_id in vehicle_tracker.track_speeds:
                    speeds = vehicle_tracker.track_speeds[track_id]
                    # La velocidad ya está en px/s, solo necesitamos el promedio.
                    avg_speed = np.mean(speeds) if speeds else 0
                    
                    cv2.putText(
                        frame_with_tracking, 
                        f"ID: {track_id} Vel: {avg_speed:.1f} px/s", 
                        (int(ltrb[0]), int(ltrb[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2
                    )

            out.write(frame_with_tracking)
        
        self.cap.release()
        out.release()
        return vehicle_tracker

# --- Las funciones de guardado y análisis se mantienen, están bien ---
def save_vehicle_data(tracker, output_dir='vehicle_data'):
    os.makedirs(output_dir, exist_ok=True)
    positions_file = os.path.join(output_dir, 'vehicle_positions.csv')
    with open(positions_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Vehicle_ID', 'Frame', 'X', 'Y'])
        for track_id, positions in tracker.track_history.items():
            for frame, (x, y) in enumerate(positions):
                writer.writerow([track_id, frame, x, y])
    
    speeds_file = os.path.join(output_dir, 'vehicle_speeds.csv')
    with open(speeds_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Vehicle_ID', 'Frame', 'Speed_Pixels_Per_Second'])
        for track_id, speeds in tracker.track_speeds.items():
            for frame, speed in enumerate(speeds):
                writer.writerow([track_id, frame, speed])

def obtener_frame_base(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def analizar_datos_vehiculos(output_dir, video_path):
    os.makedirs(output_dir, exist_ok=True)
    positions_file = os.path.join(output_dir, 'vehicle_positions.csv')
    speeds_file = os.path.join(output_dir, 'vehicle_speeds.csv')
    
    if not os.path.exists(positions_file) or not os.path.exists(speeds_file):
        st.error(f"Archivos CSV no encontrados en {output_dir}")
        return []

    positions_df = pd.read_csv(positions_file)
    speeds_df = pd.read_csv(speeds_file)
    
    if speeds_df.empty or positions_df.empty:
        st.warning("No se detectaron vehículos o no hay suficientes datos para generar gráficos.")
        return []

    generated_images = []

    # Gráfico de Velocidades
    plt.figure(figsize=(15, 7))
    sns.lineplot(data=speeds_df, x='Frame', y='Speed_Pixels_Per_Second', hue='Vehicle_ID', palette='viridis')
    plt.title('Velocidades de Vehículos (Píxeles/Segundo)')
    plt.xlabel('Frame')
    plt.ylabel('Velocidad (Píxeles/Segundo)')
    plt.legend(title='Vehículo ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    speeds_plot_path = os.path.join(output_dir, 'vehicle_speeds_plot.png')
    plt.savefig(speeds_plot_path)
    plt.close()
    generated_images.append(speeds_plot_path)

    frame_base = obtener_frame_base(video_path)
    if frame_base is not None:
        frame_base_rgb = cv2.cvtColor(frame_base, cv2.COLOR_BGR2RGB)
        merged_df = pd.merge(positions_df, speeds_df, on=['Vehicle_ID', 'Frame'])
        
        # Mapa de Calor de Dispersión
        plt.figure(figsize=(16, 10))
        plt.imshow(frame_base_rgb, alpha=0.5, aspect='auto', extent=[0, frame_base.shape[1], frame_base.shape[0], 0])
        scatter = plt.scatter(merged_df['X'], merged_df['Y'], c=merged_df['Speed_Pixels_Per_Second'], cmap='YlOrRd', alpha=0.7, s=30)
        plt.colorbar(scatter, label='Velocidad (Píxeles/Segundo)')
        plt.title('Mapa de Calor de Velocidades Vehiculares')
        plt.xlabel('Posición X (Píxeles)')
        plt.ylabel('Posición Y (Píxeles)')
        scatter_plot_path = os.path.join(output_dir, 'velocidad_heatmap_scatter.png')
        plt.savefig(scatter_plot_path, dpi=200)
        plt.close()
        generated_images.append(scatter_plot_path)

    return generated_images

# --- Funciones de Streamlit ---
def procesar_video_streamlit(uploaded_video, confidence_threshold, max_age):
    MODEL_PATH = "models/modelo_vista_superior.pt"
    
    # Usar archivos temporales para entrada y salida
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_in:
        tmp_in.write(uploaded_video.getvalue())
        video_in_path = tmp_in.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_out:
        video_out_path = tmp_out.name
    
    output_dir = tempfile.mkdtemp()
    
    model = YOLO(MODEL_PATH)
    video_processor = VideoProcessor(video_in_path, max_resolution=1280)
    
    tracker = video_processor.procesar_video(model, video_out_path, confidence_threshold, max_age)
    save_vehicle_data(tracker, output_dir)
    
    os.remove(video_in_path) # Limpiar archivo de entrada
    
    return output_dir, video_out_path


# Corrección 2: Dejar solo UNA definición de main_streamlit
def main_streamlit():
    if 'processed' not in st.session_state:
        st.session_state.processed = False
        st.session_state.output_dir = None
        st.session_state.video_out_path = None

    st.title("Análisis de Vehículos en Video 🚗💨")
    st.sidebar.header("Configuración")

    uploaded_video = st.sidebar.file_uploader("Subir Video", type=['mp4', 'avi', 'mov'])
    confidence_threshold = st.sidebar.slider("Umbral de Confianza", 0.1, 1.0, 0.75, 0.05, help="Detecta solo vehículos con una confianza superior a este valor.")
    max_age = st.sidebar.slider("Máxima Edad de Track", 1, 15, 7, help="Número de frames que un vehículo puede desaparecer antes de que se le asigne un nuevo ID.")

    if st.sidebar.button("Procesar Video") and uploaded_video:
        with st.spinner('Analizando el video, esto puede tardar unos minutos...'):
            try:
                output_dir, video_out_path = procesar_video_streamlit(uploaded_video, confidence_threshold, max_age)
                st.session_state.processed = True
                st.session_state.output_dir = output_dir
                st.session_state.video_out_path = video_out_path
                st.success("¡Procesamiento completado!")
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
                st.session_state.processed = False

    if st.session_state.processed:
        st.header("Resultados del Análisis")
        
        # Mostrar video procesado
        st.subheader("Video con Tracking")
        with open(st.session_state.video_out_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)
            st.download_button("Descargar Video Procesado", video_bytes, "video_con_tracking.mp4", "video/mp4")
        
        # Generar y mostrar gráficos
        st.subheader("Gráficos de Análisis")
        generated_images = analizar_datos_vehiculos(st.session_state.output_dir, st.session_state.video_out_path)
        
        if generated_images:
            for img_path in generated_images:
                st.image(img_path, use_column_width=True)
        
        # Descargar datos CSV
        st.subheader("Descargar Datos")
        col1, col2 = st.columns(2)
        with col1:
            with open(os.path.join(st.session_state.output_dir, 'vehicle_positions.csv'), 'rb') as f:
                st.download_button("Descargar Posiciones (CSV)", f, 'vehicle_positions.csv', 'text/csv')
        with col2:
            with open(os.path.join(st.session_state.output_dir, 'vehicle_speeds.csv'), 'rb') as f:
                st.download_button("Descargar Velocidades (CSV)", f, 'vehicle_speeds.csv', 'text/csv')


if __name__ == "__main__":
    main_streamlit()