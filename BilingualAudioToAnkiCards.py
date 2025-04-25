#!/usr/bin/env python3
import whisper
import os
import re
import csv
import unicodedata
import argparse
import json
import subprocess
import pickle
from pathlib import Path
from difflib import SequenceMatcher
import time
import concurrent.futures
import shutil
import tempfile  # Para archivos temporales

# --- Constantes ---
CHECKPOINT_DIR = ".anki_checkpoints_silence"
SUMMARY_FILE = "process_summary_silence.json"
MIN_SIMILARITY_THRESHOLD = 0.65
# --- Constantes para Detección de Silencio ---
# Ajustar estos valores según el audio:
LONG_SILENCE_THRESHOLD_S = 2.5  # Duración mínima para considerar un silencio largo (delimitador de bloque)
SILENCE_NOISE_TOLERANCE_DB = -30  # Nivel de dB por debajo del cual se considera silencio
# --- Nueva constante para padding ---
AUDIO_END_PADDING_S = 0.5  # Padding adicional al final del segmento
# --- Nueva constante para duración mínima de bloque ---
MIN_BLOCK_DURATION_S = 3.0  # Duración mínima de un bloque en segundos

# --- Funciones de Normalización y Utilidades ---
def normalize_filename(text):
    """Normaliza el texto para usarlo como nombre de archivo."""
    if isinstance(text, Path):
        text = str(text)
    text = unicodedata.normalize('NFKD', str(text)).encode('ASCII', 'ignore').decode('ASCII')
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[-\s]+', '_', text)
    # Limitar longitud para evitar problemas en algunos sistemas de archivos
    return text[:50]

def normalize_text(text):
    """Normaliza el texto para comparación (minúsculas, sin puntuación, espacios simples)."""
    if not text:
        return ""
    text = text.lower()
    # Permitir apóstrofes en inglés (ej. "don't"), pero quitar otros signos
    text = re.sub(r"[^\w\s']", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_similarity(text1, text2):
    """Calcula similitud entre dos textos usando SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    # Evitar división por cero
    if not norm_text1 or not norm_text2:
        return 0.0
    return SequenceMatcher(None, norm_text1, norm_text2).ratio()

# --- Funciones de Checkpoint ---
def create_checkpoint_dir():
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

def save_checkpoint(data, checkpoint_name):
    create_checkpoint_dir()
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pkl")
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Checkpoint guardado: {checkpoint_name}")
        return checkpoint_path
    except Exception as e:
        print(f"⚠️ Error al guardar checkpoint {checkpoint_name}: {e}")
        return None

def load_checkpoint(checkpoint_name):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pkl")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Checkpoint cargado: {checkpoint_name}")
            return data
        except Exception as e:
            print(f"⚠️ Error al cargar checkpoint {checkpoint_name}: {e}. Ignorando checkpoint.")
            return None
    return None

def clear_checkpoints():
    if os.path.exists(CHECKPOINT_DIR):
        try:
            shutil.rmtree(CHECKPOINT_DIR)
            print("✓ Checkpoints temporales eliminados")
        except Exception as e:
            print(f"⚠️ Error al eliminar checkpoints: {e}")

def save_summary(summary_data):
    try:
        with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Resumen del proceso guardado en {SUMMARY_FILE}")
    except Exception as e:
        print(f"⚠️ Error al guardar resumen: {e}")

# --- Función: Detectar Silencios Largos ---
def detect_long_silences(audio_file_path, duration_threshold, noise_db):
    """
    Usa ffmpeg silencedetect para encontrar silencios largos en el archivo de audio.
    Devuelve una lista de tuplas (start_time, end_time) de los silencios detectados.
    """
    print(f"\n--- Detectando silencios largos (>{duration_threshold}s, <{noise_db}dB) en {audio_file_path} ---")
    checkpoint_name = f"silences_{normalize_filename(audio_file_path.stem)}"
    cached_silences = load_checkpoint(checkpoint_name)
    if cached_silences:
        print("--- Usando silencios detectados cacheados ---")
        return cached_silences

    silences = []
    command = [
        'ffmpeg', '-i', str(audio_file_path),
        '-af', f'silencedetect=noise={noise_db}dB:d={duration_threshold}',
        '-f', 'null', '-'
    ]
    try:
        # Ejecutar ffmpeg y capturar stderr
        result = subprocess.run(command, capture_output=True, text=True, check=False)  # No usar check=True aquí
        output_lines = result.stderr.splitlines()

        silence_start = None
        for line in output_lines:
            if 'silencedetect' in line:
                start_match = re.search(r'silence_start:\s*([\d\.]+)', line)
                end_match = re.search(r'silence_end:\s*([\d\.]+)[\s|]*silence_duration:\s*([\d\.]+)', line)  # Capturar tambien duracion

                if start_match:
                    silence_start = float(start_match.group(1))
                    # print(f"  Detected silence start: {silence_start:.3f}") # Debug

                if end_match and silence_start is not None:
                    silence_end = float(end_match.group(1))
                    duration = float(end_match.group(2))
                    # Doble check por si acaso la duración reportada es menor al threshold (ffmpeg a veces es impreciso)
                    if duration >= duration_threshold * 0.95:  # Un pequeño margen de tolerancia
                        print(f"  -> Silencio Largo Detectado: Inicio={silence_start:.3f}s, Fin={silence_end:.3f}s, Duración={duration:.3f}s")
                        silences.append((silence_start, silence_end))
                    else:
                        print(f"  (Silencio ignorado, duración {duration:.3f}s por debajo del umbral efectivo)")

                    silence_start = None  # Reset para el próximo silencio

    except FileNotFoundError:
        print("❌ Error: ffmpeg no encontrado. Asegúrate de que está instalado y en el PATH.")
        return None
    except Exception as e:
        print(f"❌ Error al ejecutar ffmpeg para silencedetect: {e}")
        return None

    # Ordenar silencios por tiempo de inicio
    silences.sort(key=lambda x: x[0])

    if not silences:
        print("⚠️ Advertencia: No se detectaron silencios largos. El audio podría no tener la estructura esperada.")
        # Podríamos tratar todo el audio como un solo bloque en este caso
        # O devolver una lista vacía y manejarlo después.
    else:
        save_checkpoint(silences, checkpoint_name)

    return silences

# --- Función: Definir Bloques basado en Silencios (MODIFICADA) ---
def define_audio_blocks(silences, total_duration):
    """
    Define los bloques de audio entre los silencios largos detectados.
    Combina bloques cortos (< MIN_BLOCK_DURATION_S) con el siguiente bloque.
    Devuelve una lista de tuplas (start_time, end_time) para cada bloque.
    """
    print(f"\n--- Definiendo bloques de audio basados en silencios largos (mínimo {MIN_BLOCK_DURATION_S}s por bloque) ---")
    
    # Crear bloques iniciales basados en silencios
    raw_blocks = []
    last_block_end = 0.0

    for silence_start, silence_end in silences:
        block_start = last_block_end
        block_end = silence_start  # El bloque termina donde empieza el silencio largo
        
        # El bloque debe tener duración > 0
        if block_end > block_start:
            raw_blocks.append((block_start, block_end, silence_end))  # Guardamos también el fin del silencio
        
        # El próximo bloque comenzará después de que termine este silencio
        last_block_end = silence_end

    # Añadir el último bloque desde el final del último silencio hasta el final del audio
    if total_duration > last_block_end:
        raw_blocks.append((last_block_end, total_duration, total_duration))  # No hay silencio después del último bloque
        
    # Fase 2: Combinar bloques cortos con el siguiente
    final_blocks = []
    skip_next = False
    
    for i in range(len(raw_blocks)):
        if skip_next:
            skip_next = False
            continue
            
        block_start, block_end, silence_end = raw_blocks[i]
        block_duration = block_end - block_start
        
        # Verificar si este bloque es demasiado corto Y no es el último bloque
        if block_duration < MIN_BLOCK_DURATION_S and i < len(raw_blocks) - 1:
            # Combinar con el siguiente bloque
            next_block_start, next_block_end, next_silence_end = raw_blocks[i+1]
            combined_block = (block_start, next_block_end)
            combined_duration = next_block_end - block_start
            
            print(f"  -> Bloque Combinado: Inicio={block_start:.3f}s, Fin={next_block_end:.3f}s, Duración={combined_duration:.3f}s")
            print(f"     (Combinado bloque corto {block_start:.3f}s-{block_end:.3f}s [{block_duration:.3f}s] con siguiente bloque)")
            
            final_blocks.append(combined_block)
            skip_next = True  # Saltar el próximo bloque ya que lo combinamos con este
        
        else:
            # Este bloque es lo suficientemente largo o es el último
            if block_duration < MIN_BLOCK_DURATION_S:
                print(f"  -> Bloque Final (Corto): Inicio={block_start:.3f}s, Fin={block_end:.3f}s, Duración={block_duration:.3f}s")
                print(f"     (Mantenido a pesar de ser corto por ser el último bloque)")
            else:
                print(f"  -> Bloque Definido: Inicio={block_start:.3f}s, Fin={block_end:.3f}s, Duración={block_duration:.3f}s")
            
            final_blocks.append((block_start, block_end))

    print(f"--- Total de {len(final_blocks)} bloques definidos después de combinar bloques cortos ---")
    return final_blocks

# --- Función: Obtener Duración del Audio ---
def get_audio_duration(audio_file_path):
    """Obtiene la duración total del archivo de audio usando ffprobe."""
    command = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_file_path)
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        print(f"Duración total del audio: {duration:.3f} segundos.")
        return duration
    except FileNotFoundError:
        print("❌ Error: ffprobe no encontrado. Asegúrate de que ffmpeg está instalado correctamente.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar ffprobe: {e}")
        return None
    except ValueError:
        print(f"❌ Error: No se pudo convertir la salida de ffprobe a número: '{result.stdout.strip()}'")
        return None

# --- Función: Extraer segmento de audio ---
def extract_audio_segment(input_path, output_path, start_time, end_time, adaptive_padding=False):
    """
    Extrae un segmento de audio entre start_time y end_time del archivo input_path.
    Si adaptive_padding es True, añade un padding al principio/final para evitar cortes bruscos.
    """
    try:
        # Ajustar tiempos con padding si es necesario
        if adaptive_padding:
            actual_start = max(0, start_time - 0.15)  # 150ms de padding inicio
            actual_end = end_time + AUDIO_END_PADDING_S  # Usar el padding definido para el final
        else:
            actual_start = start_time
            actual_end = end_time

        # Comando ffmpeg para extraer el segmento
        cmd = [
            'ffmpeg', '-y',
            '-i', str(input_path),
            '-ss', str(actual_start),
            '-to', str(actual_end),
            '-c:a', 'libmp3lame',
            '-q:a', '3',  # Calidad alta para tarjetas Anki
            str(output_path)
        ]
        
        # Ejecutar el comando
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"❌ Error al extraer segmento {start_time}-{end_time} a {output_path}: {e}")
        return False

# --- Función para Ejecución Concurrente de Extracción de Audio ---
def extract_audio_worker(task_tuple):
    """Función worker para ThreadPoolExecutor - extrae un segmento de audio."""
    input_file, output_file, start_time, end_time = task_tuple
    success = extract_audio_segment(input_file, output_file, start_time, end_time, adaptive_padding=True)
    return output_file, success

# --- FUNCIÓN MEJORADA: Buscar Coincidencias en Transcripciones Forzadas ---
def find_matches_in_forced_transcriptions(spanish_phrase, english_phrase, es_transcription, en_transcription, block_duration):
    """
    Busca las mejores coincidencias para español e inglés en las transcripciones forzadas.
    Para inglés, capta desde la primera coincidencia hasta el final del bloque para incluir repeticiones.
    Devuelve los tiempos relativos al bloque y los scores de similitud.
    """
    print(f"  Buscando coincidencias en transcripciones forzadas...")
    
    # Buscar mejor coincidencia en español
    best_es_segment = None
    best_es_score = 0.0
    
    for segment in es_transcription.get("segments", []):
        similarity = calculate_similarity(spanish_phrase, segment["text"])
        if similarity > best_es_score:
            best_es_score = similarity
            best_es_segment = segment
    
    # Buscar PRIMERA coincidencia en inglés que supere el umbral
    first_en_segment = None
    best_en_score = 0.0
    
    for segment in en_transcription.get("segments", []):
        similarity = calculate_similarity(english_phrase, segment["text"])
        if similarity >= MIN_SIMILARITY_THRESHOLD and similarity > best_en_score:
            best_en_score = similarity
            first_en_segment = segment
            # No rompemos el bucle para encontrar la mejor coincidencia
    
    # Verificar si encontramos coincidencias por encima del umbral
    es_match = best_es_segment if best_es_score >= MIN_SIMILARITY_THRESHOLD else None
    
    # Si encontramos coincidencias, devolver los tiempos y scores
    if es_match:
        print(f"    -> Match ES: Score={best_es_score:.2f}, RelTime={es_match['start']:.2f}-{es_match['end']:.2f}")
        print(f"       Texto: '{es_match['text']}'")
    else:
        print(f"    -> No match ES (Mejor score: {best_es_score:.2f})")
    
    # Para inglés, si encontramos una coincidencia, usamos el tiempo desde ahí hasta el final del bloque
    if first_en_segment:
        # Usar desde el inicio del primer segmento inglés hasta el final del bloque
        en_start = first_en_segment["start"]
        en_end = block_duration  # Hasta el final del bloque para captar todas las repeticiones
        
        print(f"    -> Match EN: Score={best_en_score:.2f}, Primera coincidencia: {en_start:.2f}s")
        print(f"       Texto primera coincidencia: '{first_en_segment['text']}'")
        print(f"       Capturando desde {en_start:.2f}s hasta el final del bloque ({block_duration:.2f}s)")
        
        return (es_match["start"], es_match["end"]) if es_match else None, (en_start, en_end), best_es_score, best_en_score
    else:
        print(f"    -> No match EN (Mejor score: {best_en_score:.2f})")
        return (es_match["start"], es_match["end"]) if es_match else None, None, best_es_score, best_en_score

# --- Función Principal de Procesamiento (Reestructurada) ---
def process_blocks_and_phrases(original_audio_path, csv_phrases, audio_blocks, model, output_dir_path, output_csv_path):
    """
    Itera sobre los bloques de audio y las frases del CSV, procesándolos uno por uno.
    Ahora transcribe cada bloque en ambos idiomas.
    """
    print(f"\n--- Procesando {len(audio_blocks)} bloques de audio y {len(csv_phrases)} frases CSV ---")
    if len(audio_blocks) != len(csv_phrases):
        print(f"⚠️ Advertencia: El número de bloques de audio ({len(audio_blocks)}) no coincide con el número de frases en el CSV ({len(csv_phrases)}). Se procesarán hasta el mínimo de ambos.")

    output_dir = Path(output_dir_path)
    output_csv = Path(output_csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoints para esta fase
    checkpoint_name = f"processing_{normalize_filename(original_audio_path.stem)}_{len(csv_phrases)}phrases"
    checkpoint_data = load_checkpoint(checkpoint_name)

    anki_results = checkpoint_data.get("anki_results", []) if checkpoint_data else []
    failed_phrases = checkpoint_data.get("failed_phrases", []) if checkpoint_data else []
    processed_indices = checkpoint_data.get("processed_indices", set()) if checkpoint_data else set()

    # Crear directorio temporal para bloques
    with tempfile.TemporaryDirectory(prefix="audio_blocks_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        print(f"Directorio temporal para bloques: {temp_dir_path}")

        extraction_tasks = []  # Tareas para la extracción FINAL de audio
        pending_final_extractions = {}  # Para evitar duplicados

        # Iterar sobre el mínimo de bloques y frases
        num_items_to_process = min(len(audio_blocks), len(csv_phrases))
        start_item_index = 0
        if processed_indices:
            start_item_index = max(processed_indices) + 1 if processed_indices else 0
            if start_item_index < num_items_to_process:
                print(f"--- Reanudando procesamiento desde el item #{start_item_index + 1} ---")
            else:
                print("--- Todos los items ya procesados según checkpoint ---")
                # Aún necesitamos generar CSV/Resumen, así que no salimos

        for i in range(start_item_index, num_items_to_process):
            if i in processed_indices: continue

            spanish_text, english_text = csv_phrases[i]
            block_start, block_end = audio_blocks[i]
            block_duration = block_end - block_start

            print(f"\n--- Procesando Ítem #{i+1}/{num_items_to_process} ---")
            print(f"  CSV: ES='{spanish_text}' | EN='{english_text}'")
            print(f"  BLOQUE: {block_start:.3f}s - {block_end:.3f}s (Dur: {block_duration:.3f}s)")

            # 1. Extraer bloque a archivo temporal
            temp_block_filename = temp_dir_path / f"_block_{i+1}.mp3"
            print(f"  1. Extrayendo bloque a: {temp_block_filename}...")
            # Usar -c copy si el formato original es mp3, sino re-codificar
            # Para simplificar, re-codifiquemos siempre a MP3 para Whisper
            extract_block_command = [
                'ffmpeg', '-y', '-i', str(original_audio_path),
                '-ss', str(block_start), '-to', str(block_end),
                '-c:a', 'libmp3lame', '-q:a', '4',  # Calidad decente para transcripción
                '-ar', '16000',  # Whisper prefiere 16kHz
                '-ac', '1',      # Mono
                str(temp_block_filename)
            ]
            try:
                subprocess.run(extract_block_command, check=True, capture_output=True)
            except Exception as e:
                print(f"  ❌ Error al extraer bloque temporal: {e}")
                failed_phrases.append((spanish_text, english_text, f"Error extrayendo bloque {i+1}"))
                processed_indices.add(i)
                continue  # Saltar al siguiente

            # 2. MEJORA: Transcribir el bloque temporal FORZADO EN ESPAÑOL
            print(f"  2a. Transcribiendo bloque FORZADO EN ESPAÑOL: {temp_block_filename}...")
            # Usar checkpoint para la transcripción española
            block_es_checkpoint_name = f"block_es_transcript_{normalize_filename(original_audio_path.stem)}_block{i+1}"
            es_transcription_result = load_checkpoint(block_es_checkpoint_name)

            if not es_transcription_result:
                try:
                    # Añadir prompt específico para español
                    es_prompt = "Este audio contiene una frase en español."
                    es_transcription_result = model.transcribe(
                        str(temp_block_filename),
                        verbose=False,
                        word_timestamps=True,
                        language="es",  # Forzar español
                        fp16=False,
                        initial_prompt=es_prompt
                    )
                    save_checkpoint(es_transcription_result, block_es_checkpoint_name)
                except Exception as e:
                    print(f"  ❌ Error al transcribir bloque en español: {e}")
                    failed_phrases.append((spanish_text, english_text, f"Error transcribiendo bloque {i+1} en español"))
                    processed_indices.add(i)
                    continue
            else:
                print(f"  --- Usando transcripción española de bloque cacheada ---")

            # 3. MEJORA: Transcribir el bloque temporal FORZADO EN INGLÉS
            print(f"  2b. Transcribiendo bloque FORZADO EN INGLÉS: {temp_block_filename}...")
            # Usar checkpoint para la transcripción inglesa
            block_en_checkpoint_name = f"block_en_transcript_{normalize_filename(original_audio_path.stem)}_block{i+1}"
            en_transcription_result = load_checkpoint(block_en_checkpoint_name)

            if not en_transcription_result:
                try:
                    # Añadir prompt específico para inglés
                    en_prompt = "This audio contains an English phrase."
                    en_transcription_result = model.transcribe(
                        str(temp_block_filename),
                        verbose=False,
                        word_timestamps=True,
                        language="en",  # Forzar inglés
                        fp16=False,
                        initial_prompt=en_prompt
                    )
                    save_checkpoint(en_transcription_result, block_en_checkpoint_name)
                except Exception as e:
                    print(f"  ❌ Error al transcribir bloque en inglés: {e}")
                    failed_phrases.append((spanish_text, english_text, f"Error transcribiendo bloque {i+1} en inglés"))
                    processed_indices.add(i)
                    continue
            else:
                print(f"  --- Usando transcripción inglesa de bloque cacheada ---")

            # 4. MEJORA: Buscar coincidencias en AMBAS transcripciones
            # Pasamos la duración del bloque para que la función pueda usar el tiempo completo para inglés
            es_rel_times, en_rel_times, es_score, en_score = find_matches_in_forced_transcriptions(
                spanish_text, english_text, 
                es_transcription_result, en_transcription_result,
                block_duration=block_duration
            )

            # 5. Calcular tiempos absolutos si se encontraron coincidencias
            es_found = es_rel_times is not None
            en_found = en_rel_times is not None
            
            if es_found or en_found:
                # Preparar para la extracción
                file_prefix = f"{i+1:04d}"
                spanish_filename = f"{normalize_filename(f'{file_prefix}_{spanish_text}')}.mp3"
                english_filename = f"{normalize_filename(f'{file_prefix}_{english_text}')}.mp3"
                spanish_output_final = output_dir / spanish_filename
                english_output_final = output_dir / english_filename
                
                # Si encontramos el segmento español, extraerlo
                if es_found:
                    es_abs_start = block_start + es_rel_times[0]
                    es_abs_end = block_start + es_rel_times[1]
                    print(f"  -> ES Abs: {es_abs_start:.3f}s - {es_abs_end:.3f}s")
                    
                    # Tarea para extraer ESPAÑOL
                    task_es = (str(original_audio_path), str(spanish_output_final), es_abs_start, es_abs_end)
                    if str(spanish_output_final) not in pending_final_extractions:
                        extraction_tasks.append(task_es)
                        pending_final_extractions[str(spanish_output_final)] = True
                
                # Si encontramos el segmento inglés, extraerlo (ahora con todo el rango hasta final)
                if en_found:
                    en_abs_start = block_start + en_rel_times[0]
                    en_abs_end = block_start + en_rel_times[1]  # Aquí ahora será el fin del bloque
                    print(f"  -> EN Abs: {en_abs_start:.3f}s - {en_abs_end:.3f}s (Incluye repeticiones)")
                    
                    # Tarea para extraer INGLÉS
                    task_en = (str(original_audio_path), str(english_output_final), en_abs_start, en_abs_end)
                    if str(english_output_final) not in pending_final_extractions:
                        extraction_tasks.append(task_en)
                        pending_final_extractions[str(english_output_final)] = True
                
                # Guardar resultado para Anki solo si se encontraron ambas partes o si se permite parcial
                if es_found and en_found:
                    anki_results.append({
                        "index": i,
                        "spanish_text": spanish_text,
                        "english_text": english_text,
                        "spanish_filename": spanish_filename,
                        "english_filename": english_filename,
                        "es_score": es_score,
                        "en_score": en_score,
                        "es_time_abs": (es_abs_start, es_abs_end) if es_found else None,
                        "en_time_abs": (en_abs_start, en_abs_end) if en_found else None,
                    })
                    print(f"  ✓ Ítem #{i+1} mapeado para extracción final (ambos idiomas).")
                elif es_found:  # Solo español
                    anki_results.append({
                        "index": i,
                        "spanish_text": spanish_text,
                        "english_text": english_text,
                        "spanish_filename": spanish_filename,
                        "english_filename": None,  # No hay archivo inglés
                        "es_score": es_score,
                        "en_score": en_score,
                        "es_time_abs": (es_abs_start, es_abs_end),
                        "en_time_abs": None,
                    })
                    print(f"  ✓ Ítem #{i+1} mapeado para extracción final (solo español).")
                elif en_found:  # Solo inglés
                    anki_results.append({
                        "index": i,
                        "spanish_text": spanish_text,
                        "english_text": english_text,
                        "spanish_filename": None,  # No hay archivo español
                        "english_filename": english_filename,
                        "es_score": es_score,
                        "en_score": en_score,
                        "es_time_abs": None,
                        "en_time_abs": (en_abs_start, en_abs_end),
                    })
                    print(f"  ✓ Ítem #{i+1} mapeado para extracción final (solo inglés).")
            else:
                print(f"  ❌ No se encontraron coincidencias para ES ni EN en el bloque {i+1}.")
                failed_phrases.append((spanish_text, english_text, f"No matches found in block {i+1}"))

            # Marcar como procesado y guardar checkpoint
            processed_indices.add(i)
            anki_results.sort(key=lambda x: x["index"])  # Mantener orden
            save_checkpoint({
                "anki_results": anki_results,
                "failed_phrases": failed_phrases,
                "processed_indices": processed_indices,
            }, checkpoint_name)

        # --- Ejecutar Extracción Final ---
        print(f"\n--- Iniciando extracción FINAL de {len(extraction_tasks)} segmentos ---")
        failed_extractions = []
        successful_extractions = 0
        
        if extraction_tasks:
            max_workers = min(os.cpu_count() // 2, 4)  # Limitar workers
            print(f"(Usando {max_workers} workers)")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(extract_audio_worker, task): task for task in extraction_tasks}
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        output_file, success = future.result()
                        if not success:
                            failed_extractions.append(output_file)
                        else:
                            successful_extractions += 1
                    except Exception as exc:
                        task_info = future_to_task[future]
                        print(f"⚠️ Excepción en worker para tarea {task_info[1]}: {exc}")
                        failed_extractions.append(task_info[1])

            print(f"\n--- Extracción FINAL completada ---")
            print(f"  Éxitos: {successful_extractions}")
            print(f"  Fallos: {len(failed_extractions)}")
            if failed_extractions:
                print("  Archivos fallidos:", [os.path.basename(f) for f in failed_extractions])
        else:
             print("No hay tareas de extracción final pendientes.")


    # Generar CSV final y reporte de fallos
    print("\n--- Generando archivo CSV para Anki ---")
    anki_rows = []
    anki_results.sort(key=lambda x: x["index"])
    for result in anki_results:
        es_audio_path = output_dir / result.get('spanish_filename', '') if result.get('spanish_filename') else None
        en_audio_path = output_dir / result.get('english_filename', '') if result.get('english_filename') else None
        
        # Comprobar si los archivos existen
        es_sound_tag = f"[sound:{result['spanish_filename']}]" if es_audio_path and es_audio_path.exists() else ""
        en_sound_tag = f"[sound:{result['english_filename']}]" if en_audio_path and en_audio_path.exists() else ""
        
        # Si no tenemos el archivo de español o inglés, incluir un marcador
        if not es_sound_tag:
            es_note = "(audio ES no disponible)"
        else:
            es_note = ""
            
        if not en_sound_tag:
            en_note = "(audio EN no disponible)"
        else:
            en_note = ""
        
        anki_row = f"{result['spanish_text']} {es_note}{es_sound_tag};{result['english_text']} {en_note}{en_sound_tag}"
        anki_rows.append(anki_row)

    try:
        with output_csv.open('w', encoding='utf-8', newline='') as f:
            for row in anki_rows:
                f.write(row + '\n')
        print(f"✓ Archivo Anki CSV generado: {output_csv}")
    except Exception as e:
        print(f"❌ Error al escribir el archivo Anki CSV: {e}")

    if failed_phrases:
        failed_csv_path = output_csv.parent / "failed_phrases.csv"
        try:
            with failed_csv_path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(["Spanish", "English", "Reason"])
                for es, en, reason in failed_phrases:
                    writer.writerow([es, en, reason])
            print(f"✓ Reporte de frases fallidas generado: {failed_csv_path}")
        except Exception as e:
            print(f"⚠️ Error al escribir el reporte de frases fallidas: {e}")


    # Devolver resultados para el resumen final
    return len(anki_results), len(failed_phrases), successful_extractions, len(failed_extractions)

# --- Función Main (Actualizada con nuevo parámetro) ---
def main():
    global start_time_global, model_name_global, MIN_SIMILARITY_THRESHOLD, LONG_SILENCE_THRESHOLD_S
    global SILENCE_NOISE_TOLERANCE_DB, AUDIO_END_PADDING_S, MIN_BLOCK_DURATION_S

    parser = argparse.ArgumentParser(description='Genera Anki cards con audio usando detección de silencios y procesamiento por bloques con transcripción bilingüe.')
    parser.add_argument('--audio', '-a', required=True, help='Archivo de audio de entrada')
    parser.add_argument('--csv', '-c', required=True, help='Archivo CSV con frases (Español;Inglés)')
    parser.add_argument('--output', '-o', default='anki_cards.csv', help='Archivo CSV de salida Anki')
    parser.add_argument('--dir', '-d', default='audio_clips', help='Directorio para MP3 finales')
    parser.add_argument('--model', '-m', default='medium', choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'], help='Modelo Whisper')
    parser.add_argument('--min-similarity', type=float, default=MIN_SIMILARITY_THRESHOLD, help=f'Umbral similitud (def: {MIN_SIMILARITY_THRESHOLD})')
    parser.add_argument('--silence-duration', type=float, default=LONG_SILENCE_THRESHOLD_S, help=f'Duración mínima silencio largo (segundos, def: {LONG_SILENCE_THRESHOLD_S})')
    parser.add_argument('--silence-db', type=int, default=SILENCE_NOISE_TOLERANCE_DB, help=f'Nivel ruido silencio (dB, ej: -30, def: {SILENCE_NOISE_TOLERANCE_DB})')
    parser.add_argument('--end-padding', type=float, default=AUDIO_END_PADDING_S, help=f'Padding adicional al final del audio (segundos, def: {AUDIO_END_PADDING_S})')
    # Nuevo parámetro para duración mínima de bloque
    parser.add_argument('--min-block-duration', type=float, default=MIN_BLOCK_DURATION_S, help=f'Duración mínima de un bloque (segundos, def: {MIN_BLOCK_DURATION_S})')
    parser.add_argument('--force', action='store_true', help='Forzar re-detección de silencios y re-transcripción de bloques (ignorar checkpoints)')
    parser.add_argument('--allow-partial', action='store_true', help='Permitir tarjetas parciales (solo español o solo inglés)')

    args = parser.parse_args()

    # Actualizar valores globales según los argumentos
    start_time_global = time.time()
    model_name_global = args.model
    MIN_SIMILARITY_THRESHOLD = args.min_similarity
    LONG_SILENCE_THRESHOLD_S = args.silence_duration
    SILENCE_NOISE_TOLERANCE_DB = args.silence_db
    AUDIO_END_PADDING_S = args.end_padding
    MIN_BLOCK_DURATION_S = args.min_block_duration  # Actualizar el valor del nuevo parámetro

    input_audio_path = Path(args.audio)
    input_csv_path = Path(args.csv)

    # Validar entradas
    if not input_audio_path.is_file(): 
        print(f"❌ Error: Archivo audio no encontrado: {args.audio}")
        return
    if not input_csv_path.is_file(): 
        print(f"❌ Error: Archivo CSV no encontrado: {args.csv}")
        return

    # Cargar frases CSV
    csv_phrases = []
    try:
        with input_csv_path.open('r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for i, row in enumerate(reader):
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    csv_phrases.append((row[0].strip(), row[1].strip()))
                else:
                    print(f"⚠️ Fila CSV ignorada (vacía o formato incorrecto): Fila {i+1} - {row}")
        if not csv_phrases:
             print("❌ Error: El archivo CSV está vacío o no contiene frases válidas.")
             return
    except Exception as e:
        print(f"❌ Error al leer el archivo CSV: {e}")
        return

    # 1. Detectar Silencios
    # Forzar re-detección si se usa --force
    if args.force: 
        print("Forzando recálculo - eliminando checkpoints existentes")
        clear_checkpoints() # Limpiar todos los checkpoints si se fuerza

    silences = detect_long_silences(input_audio_path, LONG_SILENCE_THRESHOLD_S, SILENCE_NOISE_TOLERANCE_DB)
    if silences is None: 
        print("❌ Fallo en detección de silencios. Abortando.")
        return

    # 2. Obtener Duración Total y Definir Bloques
    total_duration = get_audio_duration(input_audio_path)
    if total_duration is None: 
        print("❌ Fallo al obtener duración del audio. Abortando.")
        return
    
    audio_blocks = define_audio_blocks(silences, total_duration)
    if not audio_blocks: 
        print("❌ No se pudieron definir bloques de audio. Abortando.")
        return

    # Advertencia si los números no coinciden
    if len(audio_blocks) != len(csv_phrases):
         print(f"\n*** ¡ADVERTENCIA! ***")
         print(f"Número de bloques detectados ({len(audio_blocks)}) difiere del número de frases CSV ({len(csv_phrases)}).")
         print(f"Esto puede causar desajustes. Se procesarán {min(len(audio_blocks), len(csv_phrases))} ítems.")
         print(f"Revisa los parámetros de detección de silencio (--silence-duration, --silence-db) o el CSV.")
         print("Sugerencia: Prueba con diferentes valores para --silence-duration o --silence-db")
         # Podríamos añadir una opción para abortar aquí si el usuario quiere
         # input("Presiona Enter para continuar o Ctrl+C para abortar...")

    # 3. Cargar Modelo Whisper (hacerlo una vez fuera del bucle)
    print(f"\n--- Cargando modelo Whisper '{args.model}'... ---")
    try:
        model = whisper.load_model(args.model)
    except Exception as e:
        print(f"❌ Error al cargar el modelo Whisper: {e}")
        return

    # 4. Procesar Bloques y Frases
    successful_cards, failed_matchings, successful_extractions, failed_extractions = process_blocks_and_phrases(
        input_audio_path,
        csv_phrases,
        audio_blocks,
        model,
        Path(args.dir),
        Path(args.output)
    )

    # 5. Resumen Final
    print(f"\n--- Resumen Final del Proceso ---")
    print(f"  Modelo Whisper: {args.model}")
    print(f"  Umbral de similitud: {MIN_SIMILARITY_THRESHOLD}")
    print(f"  Frases en CSV: {len(csv_phrases)}")
    print(f"  Bloques de Audio Detectados: {len(audio_blocks)}")
    print(f"  Tarjetas Anki Generadas: {successful_cards}")
    print(f"  Frases/Bloques Fallidos (Coincidencia): {failed_matchings}")
    print(f"  Extracciones de Audio Exitosas: {successful_extractions}")
    print(f"  Extracciones de Audio Fallidas: {failed_extractions}")
    print(f"  Tiempo total: {time.time() - start_time_global:.2f} segundos")

    # Guardar resumen detallado en JSON
    summary = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_global)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_seconds": round(time.time() - start_time_global, 2),
        "audio_file": str(input_audio_path),
        "csv_file": str(input_csv_path),
        "anki_csv_output": args.output,
        "audio_clips_dir": args.dir,
        "whisper_model": args.model,
        "similarity_threshold": MIN_SIMILARITY_THRESHOLD,
        "silence_detection_params": {
            "duration_s": LONG_SILENCE_THRESHOLD_S, 
            "noise_db": SILENCE_NOISE_TOLERANCE_DB
        },
        "end_padding_s": AUDIO_END_PADDING_S,
        "csv_phrases_count": len(csv_phrases),
        "detected_audio_blocks": len(audio_blocks),
        "successful_cards": successful_cards,
        "failed_matching": failed_matchings,
        "successful_extractions": successful_extractions,
        "failed_extractions": failed_extractions
    }
    save_summary(summary)

    # Limpiar checkpoints al final
    clear_checkpoints()

if __name__ == "__main__":
    # Establecer valores por defecto globales iniciales
    MIN_SIMILARITY_THRESHOLD = 0.65
    LONG_SILENCE_THRESHOLD_S = 2.5
    SILENCE_NOISE_TOLERANCE_DB = -30
    AUDIO_END_PADDING_S = 0.5
    start_time_global = 0
    model_name_global = ""
    main()
