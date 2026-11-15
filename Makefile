CC ?= gcc
CFLAGS ?= -Wall -Wextra -Werror -O3 -std=c11 -Iinclude -mavx2

ifeq ($(OS),Windows_NT)
	EXE := .exe
	RM := del /Q /F
	LDFLAGS := -lm
	RUN_PACKING = $(subst /,\,$(TARGET_PACKING))
	RUN_CODEC = $(subst /,\,$(TARGET_CODEC))
else
	EXE :=
	RM := rm -f
	LDFLAGS := -lpthread -lm
	RUN_PACKING = ./$(TARGET_PACKING)
	RUN_CODEC = ./$(TARGET_CODEC)
endif

TARGET_PACKING := tests/test_ternary_packing$(EXE)
TARGET_CODEC   := tests/test_ternary_codec$(EXE)

all: test

test: $(TARGET_PACKING) $(TARGET_CODEC)
	$(RUN_PACKING)
	$(RUN_CODEC)

$(TARGET_PACKING): tests/test_ternary_packing.c include/protocol_ternary.h
	$(CC) $(CFLAGS) $< -o $@

$(TARGET_CODEC): tests/test_ternary_codec.c include/ternary_codec.h include/ternary_mt.h include/protocol_ternary.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

sanitize: tests/test_ternary_packing.c tests/test_ternary_codec.c include/protocol_ternary.h include/ternary_codec.h
	$(CC) -Wall -Wextra -Werror -O1 -g -std=c11 -Iinclude -fsanitize=address,undefined tests/test_ternary_packing.c -o tests/test_ternary_packing_san$(EXE)
	$(CC) -Wall -Wextra -Werror -O1 -g -std=c11 -Iinclude -fsanitize=address,undefined tests/test_ternary_codec.c -o tests/test_ternary_codec_san$(EXE) $(LDFLAGS)
	$(subst /,\,$(if $(filter Windows_NT,$(OS)),tests/test_ternary_packing_san$(EXE),./tests/test_ternary_packing_san$(EXE)))
	$(subst /,\,$(if $(filter Windows_NT,$(OS)),tests/test_ternary_codec_san$(EXE),./tests/test_ternary_codec_san$(EXE)))

clean:
	$(RM) $(subst /,\,$(TARGET_PACKING)) $(subst /,\,$(TARGET_CODEC)) tests/test_ternary_packing_san$(EXE) tests/test_ternary_codec_san$(EXE)

.PHONY: all test sanitize clean
