CC ?= clang
CFLAGS = -Wall -Wconversion --std=gnu23
DEBUG_FLAGS = -g -O0 -DDEBUG
RELEASE_FLAGS = -O3
LDFLAGS = 

UTIL_HEADERS = src/common.h src/debug_utils.h

ifeq ($(MODE),release)
	CFLAGS += $(RELEASE_FLAGS)
else
	CFLAGS += $(DEBUG_FLAGS)
endif

libs:

cleanlibs:

all: bin/game

clean:
	rm -rf bin/*

bin/%.o: src/%.c src/%.h $(UTIL_HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

bin/main.o: src/main.c $(UTIL_HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

bin/game: bin/main.o bin/lib.o
	$(CC) $^ -o $@
