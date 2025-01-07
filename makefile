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

CFLAGS += -Ilibs/glad/include/ -Ilibs/glfw/include/ -Ilibs/cglm/include/
LDFLAGS += libs/glad/src/glad.o libs/glfw/src/libglfw3.a libs/cglm/libcglm.a

UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
	LDFLAGS += -framework opengl -framework Cocoa -framework IOKit -framework QuartzCore
endif
# TODO: link with graphics API for Linux

libs:
	cd libs/glfw && cmake . && make
	cd libs/cglm && cmake . -DCGLM_STATIC=ON && make
	cd libs/glad && $(CC) -o src/glad.o -Iinclude -c src/glad.c

all: bin/game

clean:
	rm -rf bin/*

bin/%.o: src/%.c src/%.h $(UTIL_HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

bin/main.o: src/main.c $(UTIL_HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

bin/game: bin/main.o bin/window.o bin/shader.o bin/camera.o
	$(CC) $(LDFLAGS) $^ -o $@
