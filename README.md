# Rust Kalman Filters

This repository is the beginning of a Rust package that will implement various Kalman filters. Currently, the package includes a univariate Kalman Filter and a vector Kalman Filter struct, with plans to add more types in the future.

## Features

- **Univariate Kalman Filter**: Simple implementation for single-dimensional state estimation.
- **Vector Kalman Filter**: Handles multi-dimensional state estimation.

## Dependencies

To use this package, add the following to your `Cargo.toml`:

```toml
[dependencies]
nalgebra = "0.33.0"
rand = "0.8.5"
ndarray = "0.15.6"
ndarray-rand = "0.14"
```

## Future Plans

This package is currently in its early stages. Future plans include adding:

- Extended Kalman Filter
- Unscented Kalman Filter
- Ensemble Kalman Filter
- Adaptive Kalman Filter
