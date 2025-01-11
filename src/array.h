#include "common.h"

typedef struct type_layout {
  usize size;
  usize align;
} TypeLayout;

#define TYPE_LAYOUT(TYPE) ((TypeLayout){sizeof(TYPE), alignof(TYPE)})

typedef struct opaque_array {
  usize length;
  usize capacity;
  void *items;
} OpaqueArray;

constexpr OpaqueArray EMPTY_ARRAY = {};

OpaqueArray array_new_with_capacity(TypeLayout type, usize capacity);

void array_cleanup(TypeLayout type, OpaqueArray *array);

OpaqueArray array_from(TypeLayout type, usize length, const void *items);

OpaqueArray array_clone(TypeLayout type, OpaqueArray array);

/// Remove all the elements but keep the buffer.
void array_clear(TypeLayout type, OpaqueArray *array);

void array_reserve(TypeLayout type, OpaqueArray *array, usize additional);

void array_reserve_exact(TypeLayout type, OpaqueArray *array, usize additional);

void array_reserve_exact(TypeLayout type, OpaqueArray *array, usize additional);

void array_shrink_to_fit(TypeLayout type, OpaqueArray *array);

void array_append(TypeLayout type, OpaqueArray *array, usize length, const void *tail);

void array_push(TypeLayout type, OpaqueArray *array, const void *tail);

void *array_pop(TypeLayout type, OpaqueArray *array);

void array_insert(TypeLayout type, OpaqueArray *array, usize index, const void *item);

void array_remove(TypeLayout type, OpaqueArray *array, usize index);

#define ARRAY_DOWNCAST(TYPE, OPAQUE_ARRAY) (PTR_CAST(TYPE, (OPAQUE_ARRAY)))

#define ARRAY_UPCAST(ARRAY) (PTR_CAST(OpaqueArray, (ARRAY)))

#define DEF_ARRAY(CamelCaseName, snake_case_name, TYPE)                                                                \
  typedef struct snake_case_name {                                                                                     \
    usize length;                                                                                                      \
    usize capacity;                                                                                                    \
    TYPE *items;                                                                                                       \
  } CamelCaseName;                                                                                                     \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline CamelCaseName snake_case_name##_new_with_capacity(usize capacity) {                                    \
    return ARRAY_DOWNCAST(CamelCaseName, array_new_with_capacity(TYPE_LAYOUT(TYPE), capacity));                        \
  }                                                                                                                    \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline void snake_case_name##_cleanup(CamelCaseName *array) {                                                 \
    array_cleanup(TYPE_LAYOUT(TYPE), (OpaqueArray *)array);                                                            \
  }                                                                                                                    \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline CamelCaseName snake_case_name##_from(usize length, const TYPE *items) {                                \
    return ARRAY_DOWNCAST(CamelCaseName, array_from(TYPE_LAYOUT(TYPE), length, items));                                \
  }                                                                                                                    \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline CamelCaseName snake_case_name##_clone(CamelCaseName array) {                                           \
    return ARRAY_DOWNCAST(CamelCaseName, array_clone(TYPE_LAYOUT(TYPE), ARRAY_UPCAST(&array)));                        \
  }                                                                                                                    \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline void snake_case_name##_clear(CamelCaseName *array) {                                                   \
    array_clear(TYPE_LAYOUT(TYPE), (OpaqueArray *)array);                                                              \
  }                                                                                                                    \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline void snake_case_name##_reserve(CamelCaseName *array, usize additional) {                               \
    array_reserve(TYPE_LAYOUT(TYPE), (OpaqueArray *)array, additional);                                                \
  }                                                                                                                    \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline void snake_case_name##_reserve_exact(CamelCaseName *array, usize additional) {                         \
    array_reserve_exact(TYPE_LAYOUT(TYPE), (OpaqueArray *)array, additional);                                          \
  }                                                                                                                    \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline void snake_case_name##_shrink_to_fit(CamelCaseName *array) {                                           \
    array_shrink_to_fit(TYPE_LAYOUT(TYPE), (OpaqueArray *)array);                                                      \
  }                                                                                                                    \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline void snake_case_name##_append(CamelCaseName *array, usize length, const TYPE *restrict tail) {         \
    array_append(TYPE_LAYOUT(TYPE), (OpaqueArray *)array, length, tail);                                               \
  }                                                                                                                    \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline void snake_case_name##_push(CamelCaseName *array, TYPE item) {                                         \
    array_push(TYPE_LAYOUT(TYPE), (OpaqueArray *)array, &item);                                                        \
  }                                                                                                                    \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline TYPE snake_case_name##_pop(CamelCaseName *array) {                                                     \
    return *(TYPE *)array_pop(TYPE_LAYOUT(TYPE), (OpaqueArray *)array);                                                \
  }                                                                                                                    \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline void snake_case_name##_insert(CamelCaseName *array, usize index, TYPE item) {                          \
    array_insert(TYPE_LAYOUT(TYPE), (OpaqueArray *)array, index, &item);                                               \
  }                                                                                                                    \
                                                                                                                       \
  [[gnu::flatten]]                                                                                                     \
  static inline void snake_case_name##_remove(CamelCaseName *array, usize index) {                                     \
    array_remove(TYPE_LAYOUT(TYPE), (OpaqueArray *)array, index);                                                      \
  }
