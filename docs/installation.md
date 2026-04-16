# Installation

whisper4j is published to [Maven Central](https://central.sonatype.com/artifact/io.github.sparrowlogic/whisper4j) and [GitHub Packages](https://github.com/sparrowlogic/whisper4j/packages).

## Requirements

- **Java 26+** with `--enable-preview`
- A GGML model file (e.g., `ggml-base.en.bin` from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp/tree/main))

## Maven

```xml
<dependency>
    <groupId>com.sparrowlogic</groupId>
    <artifactId>whisper4j</artifactId>
    <version>1.0.1</version>
</dependency>
```

Configure the compiler and runtime for preview features:

```xml
<build>
    <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-compiler-plugin</artifactId>
            <version>3.14.0</version>
            <configuration>
                <release>26</release>
                <compilerArgs>
                    <arg>--enable-preview</arg>
                    <arg>--add-modules</arg>
                    <arg>jdk.incubator.vector</arg>
                </compilerArgs>
            </configuration>
        </plugin>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-jar-plugin</artifactId>
            <version>3.4.2</version>
            <configuration>
                <archive>
                    <manifest>
                        <mainClass>com.example.TranscribeExample</mainClass>
                    </manifest>
                </archive>
            </configuration>
        </plugin>
    </plugins>
</build>
```

## Gradle (Kotlin DSL)

```kotlin
dependencies {
    implementation("com.sparrowlogic:whisper4j:1.0.1")
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(26)
    }
}

tasks.withType<JavaCompile> {
    options.compilerArgs.addAll(listOf(
        "--enable-preview",
        "--add-modules", "jdk.incubator.vector"
    ))
}

tasks.withType<JavaExec> {
    jvmArgs(
        "--enable-preview",
        "--add-modules", "jdk.incubator.vector",
        "--enable-native-access=ALL-UNNAMED"
    )
}
```

## Gradle (Groovy DSL)

```groovy
dependencies {
    implementation 'com.sparrowlogic:whisper4j:1.0.1'
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(26)
    }
}

tasks.withType(JavaCompile) {
    options.compilerArgs += [
        '--enable-preview',
        '--add-modules', 'jdk.incubator.vector'
    ]
}

tasks.withType(JavaExec) {
    jvmArgs '--enable-preview',
            '--add-modules', 'jdk.incubator.vector',
            '--enable-native-access=ALL-UNNAMED'
}
```

## Running

whisper4j requires preview features, the Vector API incubator module, and native access (for FFM/Panama calls to platform BLAS) at runtime:

```bash
java --enable-preview \
     --add-modules jdk.incubator.vector \
     --enable-native-access=ALL-UNNAMED \
     -jar your-app.jar
```

If your application uses the Java module path (`--module-path`) instead of the classpath, grant native access to the whisper4j module by name:

```bash
java --enable-preview \
     --add-modules jdk.incubator.vector \
     --enable-native-access=com.sparrowlogic.whisper4j \
     -p your-app.jar:lib/* \
     -m your.module/com.example.Main
```

> Without `--enable-native-access`, whisper4j still works but prints a warning on every launch and will fail in a future Java release. On macOS, native access enables Apple Accelerate BLAS via the AMX coprocessor.

## Quick Example

```java
import com.sparrowlogic.whisper4j.WhisperModel;

import java.nio.file.Path;

public class TranscribeExample {
    public static void main(String[] args) throws Exception {
        var model = WhisperModel.load(Path.of(args[0]));
        model.transcribe(Path.of(args[1])).forEach(seg ->
            System.out.printf("[%.1f - %.1f] %s%n", seg.start(), seg.end(), seg.text()));
    }
}
```

## GitHub Packages

If using the GitHub Packages registry instead of Maven Central, add the repository:

### Maven

```xml
<repositories>
    <repository>
        <id>github</id>
        <url>https://maven.pkg.github.com/sparrowlogic/whisper4j</url>
    </repository>
</repositories>
```

### Gradle

```kotlin
repositories {
    mavenCentral()
    maven {
        url = uri("https://maven.pkg.github.com/sparrowlogic/whisper4j")
        credentials {
            username = project.findProperty("gpr.user") as String? ?: System.getenv("GITHUB_ACTOR")
            password = project.findProperty("gpr.key") as String? ?: System.getenv("GITHUB_TOKEN")
        }
    }
}
```

> GitHub Packages requires authentication even for public packages. See [GitHub docs](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-apache-maven-registry).
