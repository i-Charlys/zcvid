CC ?= gcc

ifeq ($(OS),Windows_NT)
	HOST_ARCH := $(PROCESSOR_ARCHITECTURE)
	EXE := .exe
	RM := del /Q /F
	LDFLAGS := -lm
	RUN_PACKING = $(subst /,\,$(TARGET_PACKING))
	RUN_CODEC = $(subst /,\,$(TARGET_CODEC))
	RUN_PROOFS = $(subst /,\,$(TARGET_PROOFS))
else
	HOST_ARCH := $(shell uname -m)
	EXE :=
	RM := rm -f
	LDFLAGS := -lpthread -lm
	RUN_PACKING = ./$(TARGET_PACKING)
	RUN_CODEC = ./$(TARGET_CODEC)
	RUN_PROOFS = ./$(TARGET_PROOFS)
endif

ifneq (,$(filter x86_64 AMD64 amd64,$(HOST_ARCH)))
	SIMD_FLAGS ?= -mavx2
else
	SIMD_FLAGS ?=
endif

CFLAGS ?= -Wall -Wextra -O3 -std=c11 -Iinclude $(SIMD_FLAGS)

TARGET_PACKING := tests/test_zcvid_packing$(EXE)
TARGET_CODEC   := tests/test_zcvid_codec$(EXE)
TARGET_PROOFS  := tests/test_zcvid_proofs$(EXE)
TARGET_LATENT  := tests/bench_latent_hybrid$(EXE)

all: test

test: $(TARGET_PACKING) $(TARGET_CODEC) $(TARGET_PROOFS)
	$(RUN_PACKING)
	$(RUN_CODEC)
	$(RUN_PROOFS)

$(TARGET_PACKING): tests/test_zcvid_packing.c include/protocol_zcvid.h
	$(CC) $(CFLAGS) $< -o $@

$(TARGET_CODEC): tests/test_zcvid_codec.c include/zcvid_codec.h include/zcvid_entropy.h include/zcvid_mt.h include/zcvid_zcrudp_bridge.h include/protocol_zcvid.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(TARGET_PROOFS): tests/test_zcvid_proofs.c include/zcvid_codec.h include/zcvid_entropy.h include/protocol_zcvid.h include/zcvid_gpu_cpu_dec.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

sanitize: tests/test_zcvid_packing.c tests/test_zcvid_codec.c tests/test_zcvid_proofs.c include/protocol_zcvid.h include/zcvid_codec.h include/zcvid_entropy.h include/zcvid_gpu_cpu_dec.h
	$(CC) -Wall -Wextra -O1 -g -std=c11 -Iinclude $(SIMD_FLAGS) -fsanitize=address,undefined tests/test_zcvid_packing.c -o tests/test_zcvid_packing_san$(EXE)
	$(CC) -Wall -Wextra -O1 -g -std=c11 -Iinclude $(SIMD_FLAGS) -fsanitize=address,undefined tests/test_zcvid_codec.c -o tests/test_zcvid_codec_san$(EXE) $(LDFLAGS)
	$(CC) -Wall -Wextra -O1 -g -std=c11 -Iinclude $(SIMD_FLAGS) -fsanitize=address,undefined tests/test_zcvid_proofs.c -o tests/test_zcvid_proofs_san$(EXE) $(LDFLAGS)
	$(subst /,\,$(if $(filter Windows_NT,$(OS)),tests/test_zcvid_packing_san$(EXE),./tests/test_zcvid_packing_san$(EXE)))
	$(subst /,\,$(if $(filter Windows_NT,$(OS)),tests/test_zcvid_codec_san$(EXE),./tests/test_zcvid_codec_san$(EXE)))
	$(subst /,\,$(if $(filter Windows_NT,$(OS)),tests/test_zcvid_proofs_san$(EXE),./tests/test_zcvid_proofs_san$(EXE)))

# Token vs entropy-coded size on a real picture: make bench-image IMG=photo.pgm
bench-image: tests/bench_image.c include/zcvid_codec.h include/zcvid_entropy.h include/protocol_zcvid.h
	$(CC) $(CFLAGS) $< -o tests/bench_image$(EXE) $(LDFLAGS)
	$(subst /,\,$(if $(filter Windows_NT,$(OS)),tests/bench_image$(EXE),./tests/bench_image)) $(IMG)

# Latent factorization & hybrid RD benchmark
bench-latent: $(TARGET_LATENT)
	$(subst /,\,$(if $(filter Windows_NT,$(OS)),$(TARGET_LATENT),./$(TARGET_LATENT)))

$(TARGET_LATENT): tests/bench_latent_hybrid.c include/zcvid_latent_hybrid.h include/zcvid_entropy.h include/zcvid_gpu_cpu_dec.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
ifeq ($(OS),Windows_NT)
	@for %%f in ($(subst /,\,$(TARGET_PACKING) $(TARGET_CODEC) $(TARGET_PROOFS) $(TARGET_LATENT) tests/test_zcvid_packing_san.exe tests/test_zcvid_codec_san.exe tests/test_zcvid_proofs_san.exe tests/bench_image.exe)) do @if exist "%%f" del /Q /F "%%f"
else
	$(RM) $(TARGET_PACKING) $(TARGET_CODEC) $(TARGET_PROOFS) $(TARGET_LATENT) tests/test_zcvid_packing_san tests/test_zcvid_codec_san tests/test_zcvid_proofs_san tests/bench_image
endif

.PHONY: all test sanitize bench-image bench-latent clean
