# Replacing C preprocessors

## X macros

```c++
template< template<class> class Container >
class VariantContainer {
    std::variant<Container<f32>,Container<f64>> variant;
}
```

Such Variant container can be used as such 

```c++
template<class T>
struct Field{
    std::vector<T> vec;
}

using VariantField = VariantContainer<Field>;
```

here VariantField is equivalent to a type like this : 

```c++
class VariantField {
    std::variant<Field<f32>,Field<f64>> variant;
}
```
