SHELL := /bin/bash
JAVA_OPTS := --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
CP := target/test-classes:target/classes
PYTHON := python3
VENV := .venv

.PHONY: quality format checkstyle spotless test verify install-hooks build benchmark benchmark-java benchmark-python clean

install-hooks:
	git config core.hooksPath .githooks

format:
	./mvnw spotless:apply

spotless:
	./mvnw spotless:check

checkstyle:
	./mvnw checkstyle:check

build:
	./mvnw compile test-compile -q

test:
	./mvnw test

quality: spotless checkstyle test

verify:
	./mvnw verify

## ── Performance benchmarks (JSON output) ────────────────────────────

## Run both benchmarks — each line is a JSON object
benchmark: benchmark-java benchmark-python

## whisper4j benchmark — outputs single JSON line to stdout
benchmark-java: build
	@java $(JAVA_OPTS) -cp $(CP) com.sparrowlogic.whisper4j.Benchmark 1 2>/dev/null

## Python whisper benchmark — outputs single JSON line to stdout
benchmark-python: $(VENV)/bin/activate
	@$(VENV)/bin/python tools/benchmark_python.py 2>/dev/null

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -q openai-whisper

clean:
	./mvnw clean -q
