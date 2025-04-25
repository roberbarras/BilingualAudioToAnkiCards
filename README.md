# BilingualAudioToAnkiCards
Herramienta que procesa archivos de audio bilingües para crear tarjetas de Anki automáticamente. Detecta silencios en el audio para segmentarlo, transcribe cada parte en ambos idiomas y genera tarjetas con las frases y sus clips de audio correspondientes, facilitando el estudio de idiomas mediante la escucha activa.


# BilingualAudioToAnkiCards

Herramienta que procesa archivos de audio bilingües para crear tarjetas de Anki automáticamente. Detecta silencios en el audio para segmentarlo, transcribe cada parte en ambos idiomas y genera tarjetas con las frases y sus clips de audio correspondientes, facilitando el estudio de idiomas mediante la escucha activa.

## ¿Qué es?

BilingualAudioToAnkiCards es una herramienta de línea de comandos diseñada para personas que estudian idiomas y utilizan Anki como sistema de repetición espaciada. Este script automatiza el tedioso proceso de:

1. Cortar archivos de audio que contienen pares de frases en dos idiomas
2. Transcribir y verificar las frases
3. Crear tarjetas de Anki con el audio y texto correspondiente

La herramienta está especialmente diseñada para trabajar con audios que contienen pausas (silencios) entre frases, como los que se encuentran en cursos de idiomas, audiolibros bilingües, o grabaciones propias para estudio.

## Requisitos

- Python 3.6+
- [FFmpeg](https://ffmpeg.org/download.html) instalado y en el PATH del sistema
- Bibliotecas de Python:
  - whisper (OpenAI)
  - pathlib
  - argparse
  - concurrent.futures
  - otros módulos estándar de Python

Para instalar las dependencias de Python:

```bash
pip install openai-whisper
```

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tuusuario/BilingualAudioToAnkiCards.git
cd BilingualAudioToAnkiCards
```

2. Asegúrate de que FFmpeg está instalado en tu sistema y accesible desde la línea de comandos.

3. Instala las dependencias de Python:
```bash
pip install -r requirements.txt
```

## Archivos de Entrada

1. **Archivo de audio**: Un archivo de audio (MP3, WAV, etc.) que contiene frases en dos idiomas separadas por silencios.
   - Ejemplo: "Frase en español [silencio] Phrase in English [silencio largo] Siguiente frase en español..."

2. **Archivo CSV**: Un archivo CSV con las frases en ambos idiomas, separadas por punto y coma (;).
   - Formato: `Frase en español;Phrase in English`
   - Cada línea corresponde a un par de frases consecutivas en el audio

## Salida Generada

1. **Archivos de audio recortados**: Segmentos cortos de audio para cada frase en cada idioma
   - Almacenados en el directorio especificado (por defecto: `audio_clips/`)

2. **Archivo CSV para Anki**: Un archivo CSV listo para importar en Anki
   - Formato: `Frase en español [sound:archivo_español.mp3];Phrase in English [sound:archivo_ingles.mp3]`

3. **Reporte de frases fallidas**: Un archivo CSV adicional con las frases donde hubo problemas
   - Útil para identificar qué frases necesitan procesamiento manual

4. **Resumen del proceso**: Un archivo JSON con estadísticas y detalles del procesamiento

## Cómo Funciona

1. **Detección de silencios**: Utiliza FFmpeg para encontrar los silencios largos en el archivo de audio.

2. **Segmentación en bloques**: Divide el audio en bloques basándose en los silencios detectados.
   - Los bloques demasiado cortos se combinan automáticamente con el siguiente bloque.

3. **Transcripción de bloques**: Utiliza el modelo Whisper para transcribir cada bloque en ambos idiomas (español e inglés).

4. **Coincidencia de frases**: Compara las transcripciones con las frases del CSV para encontrar coincidencias.

5. **Extracción de segmentos**: Extrae los segmentos de audio correspondientes a cada frase.

6. **Generación de tarjetas Anki**: Crea un archivo CSV que puedes importar directamente en Anki.

## Uso

Uso básico:

```bash
python BilingualAudioToAnkiCards.py --audio archivo.mp3 --csv frases.csv
```

Opciones avanzadas:

```bash
python BilingualAudioToAnkiCards.py --audio archivo.mp3 --csv frases.csv --output anki_output.csv --dir audio_clips --model medium --min-similarity 0.7 --silence-duration 2.0 --silence-db -35 --min-block-duration 3.0
```

### Parámetros

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `--audio`, `-a` | Archivo de audio de entrada | (requerido) |
| `--csv`, `-c` | Archivo CSV con frases (Español;Inglés) | (requerido) |
| `--output`, `-o` | Archivo CSV de salida para Anki | `anki_cards.csv` |
| `--dir`, `-d` | Directorio para los clips MP3 | `audio_clips` |
| `--model`, `-m` | Modelo de Whisper a utilizar | `medium` |
| `--min-similarity` | Umbral mínimo de similitud para coincidencias | `0.65` |
| `--silence-duration` | Duración mínima para considerar un silencio (segundos) | `2.5` |
| `--silence-db` | Nivel de decibelios para detectar silencio | `-30` |
| `--min-block-duration` | Duración mínima de un bloque (segundos) | `3.0` |
| `--end-padding` | Padding adicional al final del segmento (segundos) | `0.5` |
| `--force` | Forzar recálculo (ignorar checkpoints) | `False` |
| `--allow-partial` | Permitir tarjetas parciales (solo un idioma) | `False` |

## Consejos

1. **Ajustar la detección de silencios**: Si el script no detecta correctamente los bloques, ajusta `--silence-duration` y `--silence-db` según la calidad de tu audio.

2. **Verificar bloques vs. frases**: Asegúrate de que el número de bloques detectados coincida aproximadamente con el número de frases en tu CSV.

3. **Combinar bloques cortos**: Usa `--min-block-duration` para asegurarte de que todos los bloques tengan una duración mínima adecuada.

4. **Modelos de Whisper**: Para audios más difíciles, usa modelos más grandes (`large`, `large-v2`) aunque serán más lentos.

## Ejemplos

### Procesamiento básico
```bash
python BilingualAudioToAnkiCards.py --audio lecciones/leccion1.mp3 --csv lecciones/leccion1_frases.csv
```

### Ajustar detección de silencios para audio con ruido
```bash
python BilingualAudioToAnkiCards.py --audio podcast_bilingue.mp3 --csv transcripcion.csv --silence-db -25 --silence-duration 1.8
```

### Usar modelo más preciso y permitir tarjetas parciales
```bash
python BilingualAudioToAnkiCards.py --audio curso_avanzado.mp3 --csv curso_frases.csv --model large --allow-partial
```

## Solución de problemas

- **Demasiados/pocos bloques detectados**: Ajusta `--silence-duration` y `--silence-db`.
- **Coincidencias incorrectas**: Aumenta el valor de `--min-similarity`.
- **Extracciones de audio cortadas**: Aumenta `--end-padding`.
- **Proceso lento**: Usa un modelo de Whisper más pequeño (`base` o `small`).

## Licencia

[MIT License](LICENSE)
