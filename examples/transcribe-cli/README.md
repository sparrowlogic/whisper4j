# transcribe-cli

Minimal CLI that transcribes a WAV file and outputs a JSONL stream of segments.

## Output Format

Each line is a JSON object:

```jsonl
{"id":1,"start":0.00,"end":30.00,"text":"Hello world"}
{"id":2,"start":30.00,"end":55.20,"text":"This is a test"}
```

## Build & Run with Docker

```bash
# 1. Install whisper4j to your local Maven repo
cd /path/to/whisper4j
./mvnw install -DskipTests -Dcheckstyle.skip -Dspotless.check.skip

# 2. Stage the artifact for Docker
mkdir -p examples/transcribe-cli/local-repo/io/github/sparrowlogic/whisper4j/1.0.1
cp ~/.m2/repository/io/github/sparrowlogic/whisper4j/1.0.1/* \
   examples/transcribe-cli/local-repo/io/github/sparrowlogic/whisper4j/1.0.1/

# 3. Build the image
cd examples/transcribe-cli
docker build -t transcribe-cli .

# 4. Run with a model and audio file
docker run --rm \
  -v /path/to/model:/data/model \
  -v /path/to/audio.wav:/data/audio.wav \
  transcribe-cli /data/model /data/audio.wav
```

## Build & Run without Docker

```bash
# Requires Java 26+ and Maven 3.9+
mvn package -DskipTests

java --enable-preview \
     --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     -cp target/transcribe-cli-1.0.0.jar:target/lib/* \
     com.example.TranscribeExample /path/to/model /path/to/audio.wav
```
