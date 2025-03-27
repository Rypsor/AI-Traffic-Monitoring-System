import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import random
import csv

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class VehicleTracker:
    def __init__(self, max_age=7, fps=None):
        self.tracker = DeepSort(max_age=max_age)
        self.track_history = {}  # Historial de posiciones por track
        self.track_speeds = {}   # Velocidades de los tracks
        self.fps = fps  # frames por segundo del video
    
    def update(self, frame, detections):
        # Actualizar tracks
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Procesar cada track
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            
            # Obtener coordenadas del centro del bounding box
            ltrb = track.to_ltrb()
            center_x = (ltrb[0] + ltrb[2]) / 2
            center_y = (ltrb[1] + ltrb[3]) / 2
            
            # Inicializar historial si es un nuevo track
            if track_id not in self.track_history:
                self.track_history[track_id] = []
                self.track_speeds[track_id] = []
            
            # Agregar posición actual al historial
            self.track_history[track_id].append((int(center_x), int(center_y)))
            
            # Calcular velocidad (si hay al menos 2 puntos)
            if len(self.track_history[track_id]) > 1:
                # Calcular distancia entre los últimos dos puntos
                last_point = self.track_history[track_id][-1]
                prev_point = self.track_history[track_id][-2]
                
                # Distancia en píxeles
                pixel_distance = np.sqrt(
                    (last_point[0] - prev_point[0])**2 + 
                    (last_point[1] - prev_point[1])**2
                )
                
                # Calcular velocidad en píxeles/segundo
                if self.fps is not None:
                    # Velocidad = distancia / tiempo
                    # Tiempo entre frames = 1/fps
                    speed_pixels_per_second = pixel_distance * self.fps
                    self.track_speeds[track_id].append(speed_pixels_per_second)
                else:
                    # Si no se conoce el FPS, usar distancia por frame
                    self.track_speeds[track_id].append(pixel_distance)
        
        return tracks

class VideoProcessor:
    def __init__(self, video_path, max_resolution=1280):
        # Abrir video y obtener propiedades
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError("No se pudo abrir el video")
        
        # Propiedades originales
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Obtener FPS del video
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Calcular nueva resolución
        self.new_width, self.new_height = self.calcular_nueva_resolucion(max_resolution)
        
        # Determinar si es necesario reducir FPS
        self.frame_skip = self.calcular_frame_skip()
    
    def calcular_nueva_resolucion(self, max_resolution):
        # Si la resolución es menor que el máximo, mantener original
        if max(self.original_width, self.original_height) <= max_resolution:
            return self.original_width, self.original_height
        
        # Calcular factor de escala
        if self.original_width > self.original_height:
            scale = max_resolution / self.original_width
        else:
            scale = max_resolution / self.original_height
        
        new_width = int(self.original_width * scale)
        new_height = int(self.original_height * scale)
        
        # Asegurar que sea par (requerido por algunos codecs)
        new_width = new_width if new_width % 2 == 0 else new_width + 1
        new_height = new_height if new_height % 2 == 0 else new_height + 1
        
        return new_width, new_height
    
    def calcular_frame_skip(self):
        # Si es mayor a 30 fps, tomar frames intercalados
        if self.original_fps > 35:  # Margen de tolerancia
            return 2  # Tomar cada segundo frame
        return 1  # Procesar todos los frames
    
    def procesar_video(self, model, confidence_threshold=0.5, max_age=3):
        # Inicializar VehicleTracker con FPS
        vehicle_tracker = VehicleTracker(
            max_age=max_age, 
            fps=self.original_fps
        )
        
        # Configurar escritura de video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            'output_video_tracking_optimizado.mp4', 
            fourcc, 
            self.original_fps / self.frame_skip, 
            (self.new_width, self.new_height)
        )
        
        # Colores para tracks
        track_colors = {}
        
        # Función para generar color único
        def get_track_color(track_id):
            if track_id not in track_colors:
                track_colors[track_id] = (
                    random.randint(50, 200), 
                    random.randint(50, 200), 
                    random.randint(50, 200)
                )
            return track_colors[track_id]
        
        # Contadores
        frame_count = 0
        processed_frame_count = 0
        
        while True:
            # Leer frame
            ret, frame = self.cap.read()
            
            # Verificar si se pudo leer el frame
            if not ret:
                break
            
            # Saltar frames si es necesario
            frame_count += 1
            if frame_count % self.frame_skip != 0:
                continue
            
            # Redimensionar frame
            frame_resized = cv2.resize(frame, (self.new_width, self.new_height))
            
            # Realizar detección con YOLO
            results = model(frame_resized, conf=confidence_threshold)
            
            # Preparar detecciones para tracker
            detections_for_tracker = []
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Coordenadas de la caja
                    x1, y1, x2, y2 = box.xyxy[0]
                    
                    # Convertir a formato [x, y, ancho, alto]
                    bbox = [
                        int(x1), 
                        int(y1), 
                        int(x2 - x1),  # ancho
                        int(y2 - y1)   # alto
                    ]
                    
                    # Confianza
                    conf = box.conf[0]
                    
                    # Añadir detección para tracking
                    detections_for_tracker.append((bbox, conf, "vehicle"))
            
            # Actualizar tracks y obtener trazado
            tracks = vehicle_tracker.update(frame_resized, detections_for_tracker)
            
            # Copiar frame para dibujar
            frame_with_tracking = frame_resized.copy()
            
            # Dibujar trazados y tracks
            for track_id, history in vehicle_tracker.track_history.items():
                # Color del track
                color = get_track_color(track_id)
                
                # Dibujar trazado
                if len(history) > 1:
                    for i in range(1, len(history)):
                        cv2.line(
                            frame_with_tracking, 
                            history[i-1], 
                            history[i], 
                            color, 
                            2
                        )
                
                # Dibujar último punto del track
                if history:
                    cv2.circle(
                        frame_with_tracking, 
                        history[-1], 
                        5, 
                        color, 
                        -1
                    )
            
            # Dibujar tracks actuales
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                color = get_track_color(track_id)
                
                # Obtener coordenadas del track
                ltrb = track.to_ltrb()
                
                # Dibujar rectángulo
                cv2.rectangle(
                    frame_with_tracking, 
                    (int(ltrb[0]), int(ltrb[1])), 
                    (int(ltrb[2]), int(ltrb[3])), 
                    color, 
                    2
                )
                
                # Calcular velocidad promedio
                if track_id in vehicle_tracker.track_speeds:
                    speeds = vehicle_tracker.track_speeds[track_id]
                    avg_speed = np.mean(speeds) * (self.original_fps / self.frame_skip) if speeds else 0
                    
                    # Mostrar velocidad
                    cv2.putText(
                        frame_with_tracking, 
                        f"ID: {track_id} Speed: {avg_speed:.2f} px/s", 
                        (int(ltrb[0]), int(ltrb[1]) - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2
                    )
            
            # Escribir frame con tracking
            out.write(frame_with_tracking)
            
            # Incrementar contador de frames procesados
            processed_frame_count += 1
        
        # Liberar recursos
        self.cap.release()
        out.release()
        
        return vehicle_tracker

# Funciones de guardado y análisis (las mismas que en el script original)
def save_vehicle_data(tracker, output_dir='vehicle_data'):
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar historial de posiciones
    positions_file = os.path.join(output_dir, 'vehicle_positions.csv')
    with open(positions_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Vehicle_ID', 'Frame', 'X', 'Y'])
        
        for track_id, positions in tracker.track_history.items():
            for frame, (x, y) in enumerate(positions):
                writer.writerow([track_id, frame, x, y])
    
    # Guardar datos de velocidad
    speeds_file = os.path.join(output_dir, 'vehicle_speeds.csv')
    with open(speeds_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Vehicle_ID', 'Frame', 'Speed_Pixels_Per_Second'])
        
        for track_id, speeds in tracker.track_speeds.items():
            for frame, speed in enumerate(speeds):
                writer.writerow([track_id, frame, speed])

# Resto del código de Streamlit (el que proporcioné en el mensaje anterior)
# ... (pegar el código de Streamlit que estaba en el mensaje previo)










def procesar_video_streamlit(uploaded_video):
    """
    Procesar video subido en Streamlit
    """
    # Ruta del modelo predefinido
    MODEL_PATH = "models/modelo_vista_superior.pt"

    # Guardar archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_video.getvalue())
        video_path = tmpfile.name

    # Crear directorio para datos
    output_dir = 'vehicle_data'
    os.makedirs(output_dir, exist_ok=True)

    # Cargar modelo YOLO
    model = YOLO(MODEL_PATH)

    # Inicializar procesador de video
    video_processor = VideoProcessor(video_path, max_resolution=1280)

    # Procesar video
    tracker = video_processor.procesar_video(
        model, 
        confidence_threshold=0.75,
        max_age=7
    )

    # Guardar datos de vehículos
    save_vehicle_data(tracker, output_dir)

    # Verificar si el video de tracking fue generado
    tracking_video_path = 'output_video_tracking_optimizado.mp4'
    if not os.path.exists(tracking_video_path):
        st.error("No se generó el video de tracking")
        return None

    # Verificar tamaño del video
    video_size = os.path.getsize(tracking_video_path)
    if video_size == 0:
        st.error("El video de tracking está vacío")
        return None

    return output_dir



def obtener_frame_base(video_path):
    """
    Obtener un frame base del video sin vehículos
    """
    import cv2
    import numpy as np
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    
    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        st.error(f"No se pudo abrir el video: {video_path}")
        return None
    
    # Obtener propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Si no hay frames, retornar None
    if total_frames == 0:
        st.error("El video no contiene frames")
        cap.release()
        return None
    
    # Intentar leer varios frames
    frame_candidates = []
    
    for _ in range(min(10, total_frames)):  # Intentar con los primeros 10 frames
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue
        
        frame_candidates.append(frame)
    
    cap.release()
    
    # Si no se encontraron frames válidos
    if not frame_candidates:
        st.error("No se pudieron leer frames del video")
        return None
    
    # Seleccionar un frame (por ejemplo, el primero o el del medio)
    selected_frame = frame_candidates[len(frame_candidates) // 2]
    
    # Verificar que el frame no esté vacío
    if selected_frame is None or selected_frame.size == 0:
        st.error("Frame seleccionado está vacío")
        return None
    
    return selected_frame

def analizar_datos_vehiculos(output_dir='vehicle_data'):
    """
    Genera análisis y visualizaciones de los datos de vehículos
    
    :param output_dir: Directorio con los datos CSV
    """
    # Asegurar que el directorio de salida exista
    os.makedirs(output_dir, exist_ok=True)

    # Leer archivos
    positions_file = os.path.join(output_dir, 'vehicle_positions.csv')
    speeds_file = os.path.join(output_dir, 'vehicle_speeds.csv')
    
    # Verificar existencia de archivos
    if not os.path.exists(positions_file) or not os.path.exists(speeds_file):
        st.error(f"Archivos CSV no encontrados en {output_dir}")
        return None

    # Cargar datos
    positions_df = pd.read_csv(positions_file)
    speeds_df = pd.read_csv(speeds_file)
    
    # 1. Gráfico de Velocidades por Vehículo
    plt.figure(figsize=(15, 7))
    
    # Verificar si hay datos de velocidad
    if not speeds_df.empty:
        for vehicle_id in speeds_df['Vehicle_ID'].unique():
            vehicle_data = speeds_df[speeds_df['Vehicle_ID'] == vehicle_id]
            plt.plot(vehicle_data['Frame'], vehicle_data['Speed_Pixels_Per_Second'], 
                     label=f'Vehículo {vehicle_id}')
        
        plt.title('Velocidades de Vehículos (Píxeles/Segundo)')
        plt.xlabel('Frame')
        plt.ylabel('Velocidad (Píxeles/Segundo)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.text(0.5, 0.5, 'No hay datos de velocidad', 
                 horizontalalignment='center', 
                 verticalalignment='center')
    
    plt.tight_layout()
    
    # Guardar imagen con ruta absoluta
    speeds_plot_path = os.path.abspath(os.path.join(output_dir, 'vehicle_speeds_plot.png'))
    plt.savefig(speeds_plot_path)
    plt.close()  # Cerrar la figura para liberar memoria
    
    # Obtener frame base del video
    video_path = 'output_video_tracking_optimizado.mp4'
    frame_base = obtener_frame_base(video_path)
    
    # Verificar que se haya obtenido un frame
    if frame_base is None:
        st.error("No se pudo obtener un frame base para el mapa de calor")
        return [speeds_plot_path]  # Devolver solo el gráfico de velocidades
    
    # Convertir frame de OpenCV a formato matplotlib
    frame_base_rgb = cv2.cvtColor(frame_base, cv2.COLOR_BGR2RGB)
    
    # 2. Mapa de calor de velocidades - Método de dispersión
    plt.figure(figsize=(16, 10))
    
    # Combinar dataframes
    merged_df = pd.merge(positions_df, speeds_df, on=['Vehicle_ID', 'Frame'])
    
    # Mostrar frame base como fondo
    plt.imshow(frame_base_rgb, alpha=0.5, aspect='auto')
    
    # Crear mapa de calor con scatter plot
    scatter = plt.scatter(
        merged_df['X'], 
        merged_df['Y'], 
        c=merged_df['Speed_Pixels_Per_Second'],  # Color basado en velocidad
        cmap='YlOrRd',              # Colormap de amarillo a rojo
        alpha=0.7,                  # Transparencia
        s=30                        # Tamaño de los puntos
    )
    
    plt.title('Mapa de Calor de Velocidades Vehiculares', fontsize=16)
    plt.xlabel('Posición X (Píxeles)', fontsize=12)
    plt.ylabel('Posición Y (Píxeles)', fontsize=12)
    
    # Añadir barra de color
    plt.colorbar(scatter, label='Velocidad (Píxeles/Segundo)')
    
    # Guardar figura con ruta absoluta
    scatter_plot_path = os.path.abspath(os.path.join(output_dir, 'velocidad_heatmap_scatter.png'))
    plt.savefig(scatter_plot_path, dpi=300)
    plt.close()  # Cerrar la figura para liberar memoria
    
    # Mapa de calor de histograma
    plt.figure(figsize=(16, 10))
    
    # Mostrar frame base como fondo
    plt.imshow(frame_base_rgb, alpha=0.5, aspect='auto')
    
    sns.histplot(
        data=merged_df, 
        x='X', 
        y='Y', 
        weights='Speed_Pixels_Per_Second',  # Usar columna de velocidad
        cmap='YlOrRd',         # Colormap de amarillo a rojo
        bins=(50, 50),         # Número de bins
        cbar=True              # Mostrar barra de color
    )
    
    plt.title('Mapa de Calor de Velocidades Vehiculares (Histograma)', fontsize=16)
    plt.xlabel('Posición X (Píxeles)', fontsize=12)
    plt.ylabel('Posición Y (Píxeles)', fontsize=12)
    
    # Guardar figura con ruta absoluta
    hist_plot_path = os.path.abspath(os.path.join(output_dir, 'velocidad_heatmap_hist.png'))
    plt.savefig(hist_plot_path, dpi=300)
    plt.close()  # Cerrar la figura para liberar memoria

    # Verificar que las imágenes se hayan generado
    generated_images = [
        speeds_plot_path,
        scatter_plot_path,
        hist_plot_path
    ]

    for img_path in generated_images:
        if not os.path.exists(img_path):
            st.error(f"No se pudo generar la imagen: {img_path}")

    return generated_images


def main_streamlit():
    # Usar st.session_state para mantener el estado
    if 'processed_output_dir' not in st.session_state:
        st.session_state.processed_output_dir = None

    st.title("Análisis de Vehículos en Video")

    # Sidebar para configuración
    st.sidebar.header("Configuración")

    # Subir video
    uploaded_video = st.sidebar.file_uploader(
        "Subir Video", 
        type=['mp4', 'avi', 'mov'], 
        help="Video para análisis de vehículos"
    )

    # Parámetros de configuración
    confidence_threshold = st.sidebar.slider(
        "Umbral de Confianza", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.75, 
        step=0.05
    )

    max_age = st.sidebar.slider(
        "Máxima Edad de Track", 
        min_value=1, 
        max_value=15, 
        value=7
    )

    # Botón de procesamiento
    if st.sidebar.button("Procesar Video") and uploaded_video:
        # Mensaje de procesamiento
        with st.spinner('Procesando video...'):
            try:
                # Procesar video
                output_dir = procesar_video_streamlit(uploaded_video)

                # Guardar en session state
                st.session_state.processed_output_dir = output_dir

                # Mostrar resultados
                st.success("Procesamiento completado!")

            except Exception as e:
                st.error(f"Error en el procesamiento: {e}")

    # Mostrar resultados si ya se ha procesado un video
    if st.session_state.processed_output_dir:
        output_dir = st.session_state.processed_output_dir

        # Sección de resultados
        st.header("Resultados del Análisis")

        # Mostrar video de tracking
        st.subheader("Video de Tracking")
        tracking_video_path = 'output_video_tracking_optimizado.mp4'
        
        try:
            # Intentar abrir el video
            with open(tracking_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                
                # Verificar si hay contenido de video
                if video_bytes:
                    # Botón de descarga en lugar de reproducción
                    st.download_button(
                        label="Descargar Video Procesado", 
                        data=video_bytes,
                        file_name="video_tracking.mp4",
                        mime="video/mp4",
                        key="download_processed_video"
                    )
                else:
                    st.error("El archivo de video está vacío")
        
        except FileNotFoundError:
            st.error(f"No se encontró el archivo de video: {tracking_video_path}")
        except Exception as e:
            st.error(f"Error al procesar el video: {e}")

        # Mostrar gráficos generados
        st.subheader("Gráficos de Análisis")
        
        # Generar imágenes
        generated_images = analizar_datos_vehiculos(output_dir)
        
        if generated_images:
            # Velocidades de vehículos
            st.image(generated_images[0], caption='Velocidades de Vehículos')
            
            # Mapas de calor
            col1, col2 = st.columns(2)
            with col1:
                st.image(generated_images[1], caption='Mapa de Calor de Velocidades (Scatter)')
            with col2:
                st.image(generated_images[2], caption='Mapa de Calor de Velocidades (Histograma)')
        else:
            st.error("No se pudieron generar los gráficos")

        # Descargar archivos CSV
        st.subheader("Descargar Datos")
        
        # Botones de descarga
        col1, col2 = st.columns(2)
        with col1:
            with open(os.path.join(output_dir, 'vehicle_positions.csv'), 'rb') as f:
                st.download_button(
                    label="Descargar Posiciones",
                    data=f,
                    file_name='vehicle_positions.csv',
                    mime='text/csv'
                )
        
        with col2:
            with open(os.path.join(output_dir, 'vehicle_speeds.csv'), 'rb') as f:
                st.download_button(
                    label="Descargar Velocidades",
                    data=f,
                    file_name='vehicle_speeds.csv',
                    mime='text/csv'
                )



if __name__ == "__main__":
    main_streamlit()