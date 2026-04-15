#!/bin/bash
JAVA=/Users/jpeterson/Library/Java/JavaVirtualMachines/corretto-26/Contents/Home/bin/java
CP="target/test-classes:target/classes"
for f in /Users/jpeterson/.m2/repository/org/junit/jupiter/junit-jupiter/5.12.1/*.jar \
         /Users/jpeterson/.m2/repository/org/junit/jupiter/junit-jupiter-api/5.12.1/*.jar \
         /Users/jpeterson/.m2/repository/org/opentest4j/opentest4j/1.3.0/*.jar \
         /Users/jpeterson/.m2/repository/org/junit/platform/junit-platform-commons/1.12.1/*.jar \
         /Users/jpeterson/.m2/repository/org/apiguardian/apiguardian-api/1.1.2/*.jar \
         /Users/jpeterson/.m2/repository/org/junit/jupiter/junit-jupiter-params/5.12.1/*.jar \
         /Users/jpeterson/.m2/repository/org/junit/jupiter/junit-jupiter-engine/5.12.1/*.jar \
         /Users/jpeterson/.m2/repository/org/junit/platform/junit-platform-engine/1.12.1/*.jar; do
    CP="$CP:$f"
done
exec $JAVA --add-modules jdk.incubator.vector --enable-preview --enable-native-access=ALL-UNNAMED -cp "$CP" com.sparrowlogic.whisper4j.TestTranscribeFiles
