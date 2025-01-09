#pragma once

#include "common.h"

constexpr usize STRING_INIT_CAPACITY = 64;

typedef struct string {
  usize capacity;
  usize length;
  char *buffer;
} String;

constexpr String EMPTY_STRING = {};

String string_new_with_capacity(usize capacity);

String string_clone(String string);

/// For `length = 0`, returns an empty string.
/// For creating a `String` from a null-terminated string, use `string_from_c_string`.
/// Note that `string` is nullable if `length = 0`. This is because we're using GNU C.
String string_from(usize length, char string[length]);

#define STRING_LITERAL(S) (string_from(sizeof(S) - 1, S))

/// For initializing with string literals, use `STRING_LITERAL`.
String string_from_c_string(char *c_string);

void string_cleanup(String *string);

/// Reserve at least `additional` bytes after `length`.
/// (Would reserve more than that if deemed efficient.)
void string_reserve(String *string, usize additional);

/// Reserve exactly `additional` bytes after `length`, unless the current capacity is already enough.
void string_reserve_exact(String *string, usize additional);

void string_shrink_to_fit(String *string);

void string_push(String *string, char tail);

void string_append(String *string, usize length, const char tail[restrict length]);

/// Clear the content but retain the buffer and its capacity.
void string_clear(String *string);

/// Fuck C strings.
/// Note that the extra '\0' is past `length`.
void string_to_c_string(String *string);

[[gnu::format(printf, 3, 4)]]
void string_snprintf(String *string, usize n, const char *restrict fmt, ...);

void string_fprint(String string, FILE *restrict out);

void string_print(String string);
