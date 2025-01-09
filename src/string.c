#include "string.h"
#include <stdarg.h>

String string_new_with_capacity(usize capacity) {
  return (String){
      .capacity = capacity,
      .length = 0,
      .buffer = xalloc(char, capacity),
  };
}

String string_clone(String string) {
  return string_from(string.length, string.buffer);
}

String string_from(usize length, char string[length]) {
  return (String){
      .capacity = length,
      .length = length,
      .buffer = (length == 0) ? nullptr : memcpy(xalloc(char, length), string, length),
  };
}

String string_from_c_string(char *c_string) {
  DEBUG_ASSERT(c_string != nullptr);
  auto string = EMPTY_STRING;
  for (usize i = 0; c_string[i] != '\0'; ++i)
    string_push(&string, c_string[i]);
  return string;
}

void string_cleanup(String *string) {
  xfree(string->buffer);
  if (IS_DEBUG_MODE)
    string->buffer = nullptr;
}

static inline void realloc_the_string(String *string, usize new_capacity) {
  if (string->buffer == nullptr)
    string->buffer = xalloc(char, new_capacity);
  else
    string->buffer = xrealloc(string->buffer, char, new_capacity);
  string->capacity = new_capacity;
}

void string_reserve(String *string, usize additional) {
  auto new_capacity = string->length + additional;
  if (new_capacity > string->capacity)
    realloc_the_string(string, MAX(string->capacity * 2, new_capacity));
}

void string_reserve_exact(String *string, usize additional) {
  auto new_capacity = string->length + additional;
  if (new_capacity > string->capacity)
    realloc_the_string(string, new_capacity);
}

void string_shrink_to_fit(String *string) {
  // Why would someone ever need this bruh.
  auto new_string = string_clone(*string);
  string_cleanup(string);
  *string = new_string;
}

void string_push(String *string, char tail) {
  string_reserve(string, 1);
  string->buffer[string->length] = tail;
  ++string->length;
}

void string_append(String *string, usize tail_length, const char tail[restrict tail_length]) {
  string_reserve(string, tail_length);
  memcpy(&string->buffer[string->length], tail, tail_length);
  string->length += tail_length;
}

void string_clear(String *string) {
  string->length = 0;
}

void string_to_c_string(String *string) {
  string_reserve(string, 1);
  string->buffer[string->length] = '\0';
}

void string_fprint(String string, FILE *restrict out) {
  fwrite(string.buffer, 1, string.length, out);
}

void string_print(String string) {
  string_fprint(string, stdout);
}

[[gnu::format(printf, 3, 4)]]
void string_snprintf(String *string, usize n, const char *restrict fmt, ...) {
  string_reserve(string, n);
  va_list args;
  va_start(args, fmt);
  auto growth = (usize)vsnprintf(&string->buffer[string->length], n, fmt, args);
  va_end(args);
  string->length += growth;
}
