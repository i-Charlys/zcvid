CC ?= gcc
CFLAGS ?= -Wall -Wextra -Werror -O3 -std=c11 -Iinclude -mavx2

ifeq ($(OS),Windows_NT)
	EXE := .exe
	RM := del /Q /F
	LDFLAGS := -lm
	RUN_PACKING = $(subst /,\,$(TARGET_PACKING))
	RUN_CODEC = $(subst /,\,$(TARGET_CODEC))
	RUN_PROOFS = $(subst /,\,$(TARGET_PROOFS))
else
	EXE :=
	RM := rm -f
	LDFLAGS := -lpthread -lm
	RUN_PACKING = ./$(TARGET_PACKING)
	RUN_CODEC = ./$(TARGET_CODEC)
	RUN_PROOFS = ./$(TARGET_PROOFS)
endif

TARGET_PACKING := tests/test_ternary_packing$(EXE)
TARGET_CODEC   := tests/test_ternary_codec$(EXE)
TARGET_PROOFS  := tests/test_ternary_proofs$(EXE)

all: test

test: $(TARGET_PACKING) $(TARGET_CODEC) $(TARGET_PROOFS)
	$(RUN_PACKING)
	$(RUN_CODEC)
	$(RUN_PROOFS)

$(TARGET_PACKING): tests/test_ternary_packing.c include/protocol_ternary.h
	$(CC) $(CFLAGS) $< -o $@

$(TARGET_CODEC): tests/test_ternary_codec.c include/ternary_codec.h include/ternary_entropy.h include/ternary_mt.h include/ternary_zcrudp_bridge.h include/protocol_ternary.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(TARGET_PROOFS): tests/test_ternary_proofs.c include/ternary_codec.h include/ternary_entropy.h include/protocol_ternary.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

sanitize: tests/test_ternary_packing.c tests/test_ternary_codec.c tests/test_ternary_proofs.c include/protocol_ternary.h include/ternary_codec.h include/ternary_entropy.h
	$(CC) -Wall -Wextra -Werror -O1 -g -std=c11 -Iinclude -mavx2 -fsanitize=address,undefined tests/test_ternary_packing.c -o tests/test_ternary_packing_san$(EXE)
	$(CC) -Wall -Wextra -Werror -O1 -g -std=c11 -Iinclude -mavx2 -fsanitize=address,undefined tests/test_ternary_codec.c -o tests/test_ternary_codec_san$(EXE) $(LDFLAGS)
	$(CC) -Wall -Wextra -Werror -O1 -g -std=c11 -Iinclude -mavx2 -fsanitize=address,undefined tests/test_ternary_proofs.c -o tests/test_ternary_proofs_san$(EXE) $(LDFLAGS)
	$(subst /,\,$(if $(filter Windows_NT,$(OS)),tests/test_ternary_packing_san$(EXE),./tests/test_ternary_packing_san$(EXE)))
	$(subst /,\,$(if $(filter Windows_NT,$(OS)),tests/test_ternary_codec_san$(EXE),./tests/test_ternary_codec_san$(EXE)))
	$(subst /,\,$(if $(filter Windows_NT,$(OS)),tests/test_ternary_proofs_san$(EXE),./tests/test_ternary_proofs_san$(EXE)))

# Token vs entropy-coded size on a real picture: make bench-image IMG=photo.pgm
bench-image: tests/bench_image.c include/ternary_codec.h include/ternary_entropy.h include/protocol_ternary.h
	$(CC) $(CFLAGS) $< -o tests/bench_image$(EXE) $(LDFLAGS)
	$(subst /,\,$(if $(filter Windows_NT,$(OS)),tests/bench_image$(EXE),./tests/bench_image)) $(IMG)

clean:
ifeq ($(OS),Windows_NT)
	@for %%f in ($(subst /,\,$(TARGET_PACKING) $(TARGET_CODEC) $(TARGET_PROOFS) tests/test_ternary_packing_san.exe tests/test_ternary_codec_san.exe tests/test_ternary_proofs_san.exe tests/bench_image.exe)) do @if exist "%%f" del /Q /F "%%f"
else
	$(RM) $(TARGET_PACKING) $(TARGET_CODEC) $(TARGET_PROOFS) tests/test_ternary_packing_san tests/test_ternary_codec_san tests/test_ternary_proofs_san tests/bench_image
endif

.PHONY: all test sanitize bench-image clean
