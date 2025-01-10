#pragma once

#include <assert.h>
#include <execinfo.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STRING_LITERAL_ARG(S) sizeof(S) - 1, S

#define ARR_ARG(ARR) (sizeof(ARR) / sizeof(ARR[0])), (typeof(ARR[0]) *)(&ARR[0])

// C23's `auto` doesn't work with GNU's cleanup attribute :(
#define auto __auto_type

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;
typedef int64_t i64;
typedef int32_t i32;
typedef int16_t i16;
typedef int8_t i8;
typedef float f32;
typedef double f64;
typedef size_t usize;
typedef ssize_t isize;

static inline f32 absf(f32 f) {
  return (f >= 0.f) ? (f) : (-f);
}

#define MAX(X, Y)                                                                                                      \
  ({                                                                                                                   \
    auto x = (X);                                                                                                      \
    auto y = (Y);                                                                                                      \
    (x > y) ? x : y;                                                                                                   \
  })

#define MIN(X, Y)                                                                                                      \
  ({                                                                                                                   \
    auto x = (X);                                                                                                      \
    auto y = (Y);                                                                                                      \
    (x < y) ? x : y;                                                                                                   \
  })

#define MARK_USED(X) ({ [[maybe_unused]] auto _ = (X); })

#define PUT_ON_HEAP(X) ((typeof(X) *restrict)memcpy(malloc(sizeof(X)), &X, sizeof(X)))

#define ARR_LEN(X) ((usize)(sizeof(X) / sizeof((X)[0])))

#define REF(X) ((const typeof(X) *restrict)&(struct { typeof(X) _; }){X})

static inline void print_stacktrace() {
  void *callstack[128];
  auto frames = backtrace(callstack, 128);
  char **strs = backtrace_symbols(callstack, frames);
  for (i32 i = 0; i < frames; i++) {
    fprintf(stderr, "%s\n", strs[i]);
  }
  free(strs);
}

#ifdef DEBUG
constexpr bool IS_DEBUG_MODE = true;
#else
constexpr bool IS_DEBUG_MODE = false;
#endif

/// Assert with stacktrace on failure.
#define ASSERT(COND)                                                                                                   \
  ({                                                                                                                   \
    auto cond = COND;                                                                                                  \
    if (!cond) {                                                                                                       \
      fprintf(stderr, "[%s@%s:%d] Assertion failed: (%s) == false\n", __FUNCTION__, __FILE__, __LINE__, #COND);        \
      print_stacktrace();                                                                                              \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  })
#define ASSERT_PRINTF(COND, ...)                                                                                       \
  ({                                                                                                                   \
    auto cond = COND;                                                                                                  \
    if (!cond) {                                                                                                       \
      fprintf(stderr, "[%s@%s:%d] Assertion failed: (%s) == false\n", __FUNCTION__, __FILE__, __LINE__, #COND);        \
      fprintf(stderr, __VA_ARGS__);                                                                                    \
      print_stacktrace();                                                                                              \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  })

#define DEBUG_ASSERT(COND)                                                                                             \
  ({                                                                                                                   \
    auto cond = COND;                                                                                                  \
    if (IS_DEBUG_MODE && !cond) {                                                                                      \
      fprintf(stderr, "[%s@%s:%d] Assertion failed: (%s) == false\n", __FUNCTION__, __FILE__, __LINE__, #COND);        \
      print_stacktrace();                                                                                              \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  })
#define DEBUG_ASSERT_PRINTF(COND, ...)                                                                                 \
  ({                                                                                                                   \
    auto cond = COND;                                                                                                  \
    if (IS_DEBUG_MODE && !cond) {                                                                                      \
      fprintf(stderr, "[%s@%s:%d] Assertion failed: (%s) == false\n", __FUNCTION__, __FILE__, __LINE__, #COND);        \
      fprintf(stderr, __VA_ARGS__);                                                                                    \
      print_stacktrace();                                                                                              \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  })

#define PANIC() (fprintf(stderr, "[%s@%s:%d] PANIC\n", __FUNCTION__, __FILE__, __LINE__), print_stacktrace(), exit(1))
#define PANIC_PRINTF(...)                                                                                              \
  (fprintf(stderr, "[%s@%s:%d] PANIC\n", __FUNCTION__, __FILE__, __LINE__), fprintf(stderr, __VA_ARGS__),              \
   print_stacktrace(), exit(1))

/// Return `0` to the caller if value is `0`
#define TRY_NULL(X)                                                                                                    \
  ({                                                                                                                   \
    auto x = (X);                                                                                                      \
    if (x == 0) {                                                                                                      \
      return 0;                                                                                                        \
    }                                                                                                                  \
    x;                                                                                                                 \
  })

#define PTR_CAST(TY, X) (*(TY *)&(X))

#define FIXME(...) (printf("[%s@%s:%d] FIXME:", __FUNCTION__, __FILE__, __LINE__), printf(__VA_ARGS__), printf("\n"))

#define TODO() (printf("[%s@%s:%d] TODO\n", __FUNCTION__, __FILE__, __LINE__), print_stacktrace(), exit(1))
#define TODO_FUNCTION()                                                                                                \
  (printf("[%s@%s:%d] TODO: function not implemented\n", __FUNCTION__, __FILE__, __LINE__), print_stacktrace(), exit(1))

__attribute__((always_inline)) static inline void *xalloc_(usize len) {
  void *p = malloc(len);
  ASSERT(p != nullptr || len == 0);
  return p;
}

__attribute__((always_inline)) static inline void *xrealloc_(void *p, usize len) {
  p = realloc(p, len);
  ASSERT(p != nullptr || len == 0);
  return p;
}

__attribute__((always_inline)) static inline void xfree(void *p) {
  free(p);
}

#define xalloc(TY, COUNT) ((TY *restrict)xalloc_(sizeof(TY) * (COUNT)))
#define xrealloc(P, TY, COUNT) ((TY *)xrealloc_((P), sizeof(TY) * (COUNT)))
