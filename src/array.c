#include "array.h"

OpaqueArray array_new_with_capacity(TypeLayout type, usize capacity) {
  return (OpaqueArray){
      .length = 0,
      .capacity = capacity,
      .items = aligned_alloc(type.align, capacity * type.size),
  };
}

void array_cleanup(TypeLayout type, OpaqueArray *array) {
  MARK_USED(type);
  xfree(array->items);
}

OpaqueArray array_from(TypeLayout type, usize length, void *items) {
  OpaqueArray array = array_new_with_capacity(type, length);
  memcpy(array.items, items, length * type.size);
  return array;
}

OpaqueArray array_clone(TypeLayout type, OpaqueArray array) {
  return array_from(type, array.length, array.items);
}

/// Remove all the elements but keep the buffer.
void array_clear(TypeLayout type, OpaqueArray *array) {
  MARK_USED(type);
  array->length = 0;
}

static inline void realloc_array(TypeLayout type, OpaqueArray *array, usize new_capacity) {
  array->items = xrealloc_(array->items, new_capacity * type.size);
  array->capacity = new_capacity;
}

void array_reserve(TypeLayout type, OpaqueArray *array, usize additional) {
  auto new_capacity = array->length + additional;
  if (new_capacity > array->capacity)
    realloc_array(type, array, MAX(array->capacity * 2, new_capacity));
}

void array_reserve_exact(TypeLayout type, OpaqueArray *array, usize additional) {
  auto new_capacity = array->length + additional;
  if (new_capacity > array->capacity)
    realloc_array(type, array, new_capacity);
}

void array_shrink_to_fit(TypeLayout type, OpaqueArray *array) {
  if (array->length != array->capacity) {
    auto new_array = array_clone(type, *array);
    array_cleanup(type, array);
    *array = new_array;
  }
}

void array_append(TypeLayout type, OpaqueArray *array, usize length, const void *restrict tail) {
  array_reserve(type, array, length);
  memcpy(&array->items[array->length * type.size], tail, length * type.size);
  array->length += length;
}

void array_push(TypeLayout type, OpaqueArray *array, const void *restrict tail) {
  array_append(type, array, 1, tail);
}

void *array_pop(TypeLayout type, OpaqueArray *array) {
  --array->length;
  return &array->items[array->length * type.size];
}

void array_insert(TypeLayout type, OpaqueArray *array, usize index, const void *restrict item) {
  array_reserve(type, array, 1);
  memmove(                                    //
      &array->items[(index + 1) * type.size], //
      &array->items[index * type.size],       //
      (array->length - index) * type.size);
  memcpy(&array->items[index * type.size], item, type.size);
}

void array_remove(TypeLayout type, OpaqueArray *array, usize index) {
  memmove(                                    //
      &array->items[index * type.size],       //
      &array->items[(index + 1) * type.size], //
      (array->length - index) * type.size);
  --array->length;
}
