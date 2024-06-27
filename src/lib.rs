extern crate nalgebra as na;

use na::{DMatrix, DVector};
/// A structure representing a univariate Kalman Filter
struct KalmanFilter {
    state_transition: DMatrix<f64>,  // State transition matrix
    control_input: DMatrix<f64>,  // Control input matrix
    observation: DMatrix<f64>,  // Observation matrix
    process_covariance: DMatrix<f64>,  // Process noise covariance
    measurement_covariance: DMatrix<f64>,  // Measurement noise covariance
    error_covariance: DMatrix<f64>,  // Error covariance matrix
    x: DVector<f64>,  // State estimate vector
}

impl KalmanFilter {
    /// Creates a new instance of the Kalman Filter
    fn new(
        state_transition: DMatrix<f64>,
        control_input: Option<DMatrix<f64>>,
        observation: DMatrix<f64>,
        process_covariance: Option<DMatrix<f64>>,
        measurement_covariance: Option<DMatrix<f64>>,
        error_covariance: Option<DMatrix<f64>>,
        x0: Option<DVector<f64>>,
    ) -> Self {
        let n = state_transition.ncols();
        let m = observation.nrows();  

        KalmanFilter {
            state_transition,
            control_input: control_input.unwrap_or_else(|| DMatrix::zeros(n, 1)),
            observation,
            process_covariance: process_covariance.unwrap_or_else(|| DMatrix::identity(n, n)),
            measurement_covariance: measurement_covariance.unwrap_or_else(|| DMatrix::identity(m, m)),
            error_covariance: error_covariance.unwrap_or_else(|| DMatrix::identity(n, n)),
            x: x0.unwrap_or_else(|| DVector::zeros(n)),
        }
    }

    /// Predicts the next state
    fn predict(&mut self, u: Option<DVector<f64>>) -> DVector<f64> {
        let control_vector = control_vector.unwrap_or_else(|| DVector::zeros(self.control_input.ncols()));
        self.x = &self.state_transition * &self.x + &self.control_input * control_vector;
        self.error_covariance = &self.state_transition * &self.error_covariance * self.state_transition.transpose() + &self.process_covariance;
        self.x.clone()
    }

    /// Updates the state with a new measurement
    fn update(&mut self, z: &DVector<f64>) {
        let innovation = z - &self.observation * &self.x;  // Measurement residual
        let innovation_covariance = &self.measurement_covariance + &self.observation * &self.error_covariance * self.observation.transpose();  // Residual covariance
        let kalman_gain = &self.error_covariance * self.observation.transpose() * innovation_covariance.try_inverse().expect("Matrix is not invertible");  // Kalman gain

        self.x = &self.x + kalman_gain.clone() * innovation;
        let identity_matrix = DMatrix::identity(self.error_covariance.nrows(), self.error_covariance.ncols());
        self.error_covariance = (&identity_matrix - &kalman_gain * &self.observation) * &self.error_covariance * (&identity_matrix - &kalman_gain.clone() * &self.observation).transpose() + &kalman_gain.clone() * &self.measurement_covariance * &kalman_gain.transpose();
    }
}
