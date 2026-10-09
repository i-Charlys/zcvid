CC ?= gcc
CFLAGS ?= -Wall -Wextra -Werror -O3 -std=c11 -Iinclude

all: test

test: tests/test_ternary_packing
	./tests/test_ternary_packing

tests/test_ternary_packing: tests/test_ternary_packing.c include/protocol_ternary.h
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f tests/test_ternary_packing

.PHONY: all test clean
