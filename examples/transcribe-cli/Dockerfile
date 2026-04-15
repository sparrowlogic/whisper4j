# Stage 1: Build
FROM amazoncorretto:26 AS build
RUN yum install -y maven && yum clean all

WORKDIR /app
COPY pom.xml .
COPY src src
COPY local-repo /root/.m2/repository

RUN mvn -B package -DskipTests

# Stage 2: Run
FROM amazoncorretto:26

WORKDIR /app
COPY --from=build /app/target/transcribe-cli-1.0.0.jar app.jar
COPY --from=build /app/target/lib lib

ENTRYPOINT ["java", \
    "--enable-preview", \
    "--add-modules", "jdk.incubator.vector", \
    "--enable-native-access=ALL-UNNAMED", \
    "-cp", "app.jar:lib/*", \
    "com.example.TranscribeExample"]
# Usage: docker run -v /path/to/model:/data/model.bin -v /path/to/audio.wav:/data/audio.wav \
#        transcribe-cli /data/model.bin /data/audio.wav
