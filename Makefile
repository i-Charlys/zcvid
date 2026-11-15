CC ?= gcc
CFLAGS ?= -Wall -Wextra -Werror -O3 -std=c11 -Iinclude

ifeq ($(OS),Windows_NT)
	EXE := .exe
	RM := del /Q /F
	RUN_TARGET = $(subst /,\,$(TARGET))
else
	EXE :=
	RM := rm -f
	RUN_TARGET = ./$(TARGET)
endif

TARGET := tests/test_ternary_packing$(EXE)

all: test

test: $(TARGET)
	$(RUN_TARGET)

$(TARGET): tests/test_ternary_packing.c include/protocol_ternary.h
	$(CC) $(CFLAGS) $< -o $@

sanitize: tests/test_ternary_packing.c include/protocol_ternary.h
	$(CC) -Wall -Wextra -Werror -O1 -g -std=c11 -Iinclude -fsanitize=address,undefined $< -o tests/test_ternary_packing_san$(EXE)
	$(subst /,\,$(if $(filter Windows_NT,$(OS)),tests/test_ternary_packing_san$(EXE),./tests/test_ternary_packing_san$(EXE)))

clean:
	$(RM) $(subst /,\,$(TARGET)) tests/test_ternary_packing_san$(EXE)

.PHONY: all test sanitize clean
