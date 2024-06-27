extern crate nalgebra as na;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::{Array, Array2, Array3, Axis, s};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;

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


// Define the vector Kalman filter
struct VectorKalmanFilter {
    // State transition matrix
    state_transition: Array2<f64>,
    // Control input matrix
    control_input: Array2<f64>,
    // Observation model matrix
    observation_model: Array2<f64>,
    // Process noise covariance matrix
    process_noise_covariance: Array2<f64>,
    // Measurement noise covariance matrix
    measurement_noise_covariance: Array2<f64>,
    // State estimate vector
    state_estimate: Array2<f64>,
    // Error covariance tensor (3D array)
    error_covariance: Array3<f64>,
}

impl VectorKalmanFilter {
    // Constructor function to initialize a new KalmanFilter instance
    fn new(
        state_transition: Array2<f64>,        // State transition matrix
        control_input: Array2<f64>,           // Control input matrix
        observation_model: Array2<f64>,       // Observation model matrix
        process_noise_covariance: Array2<f64>,// Process noise covariance matrix
        measurement_noise_covariance: Array2<f64>, // Measurement noise covariance matrix
        initial_state_estimate: Array2<f64>,  // Initial state estimate
        initial_error_covariance: Array2<f64>, // Initial error covariance
        n_timesteps: usize,                   // Number of time steps for the filter
    ) -> Self {
        // Get the dimension of the state vector
        let n_dim_state = initial_state_estimate.len();
        // Initialize the error covariance tensor with zeros
        let error_covariance = Array3::<f64>::zeros((n_timesteps, n_dim_state, n_dim_state));
        
        // Create the KalmanFilter instance and set the initial error covariance
        let mut kf = Self { 
            state_transition, 
            control_input, 
            observation_model, 
            process_noise_covariance, 
            measurement_noise_covariance, 
            state_estimate: initial_state_estimate.clone(), 
            error_covariance 
        };
        // Set the initial error covariance
        kf.error_covariance.slice_mut(s![0, .., ..]).assign(&initial_error_covariance);

        // Return the initialized KalmanFilter instance
        kf
    }

    // Prediction step of the Kalman Filter
    fn predict(&mut self, control_vector: &Array2<f64>, t: usize) {
        // Predict the next state estimate
        self.state_estimate = self.state_transition.dot(&self.state_estimate) + self.control_input.dot(control_vector);
        // Predict the next error covariance
        let predicted_error_covariance = self.state_transition
            .dot(&self.error_covariance.slice(s![t, .., ..]))
            .dot(&self.state_transition.t()) + &self.process_noise_covariance;
        // Store the predicted error covariance
        self.error_covariance.slice_mut(s![t + 1, .., ..]).assign(&predicted_error_covariance);
    }

    // Update step of the Kalman Filter
    fn update(&mut self, observation: &Array2<f64>, t: usize) {
        // Compute the Kalman gain
        let kalman_gain = self.error_covariance.slice(s![t + 1, .., ..])
            .dot(&self.observation_model.t())
            .dot(&(self.observation_model
                .dot(&self.error_covariance.slice(s![t + 1, .., ..]))
                .dot(&self.observation_model.t()) + &self.measurement_noise_covariance)
                .inv().unwrap());
        
        // Update the state estimate with the measurement
        self.state_estimate = &self.state_estimate + &kalman_gain.dot(&(observation - &self.observation_model.dot(&self.state_estimate)));
        // Create an identity matrix for the state dimension
        let identity_matrix = Array2::<f64>::eye(self.state_estimate.len());
        // Update the error covariance
        let updated_error_covariance = (identity_matrix - &kalman_gain.dot(&self.observation_model))
            .dot(&self.error_covariance.slice(s![t + 1, .., ..]));
        // Store the updated error covariance
        self.error_covariance.slice_mut(s![t + 1, .., ..]).assign(&updated_error_covariance);
    }

    // Method to calculate the innovation (measurement residual)
    fn innovation(&self, observation: &Array2<f64>, t: usize) -> Array2<f64> {
        observation - &self.observation_model.dot(&self.state_estimate)
    }

    // Method to calculate the innovation covariance
    fn innovation_covariance(&self, t: usize) -> Array2<f64> {
        self.observation_model
            .dot(&self.error_covariance.slice(s![t + 1, .., ..]))
            .dot(&self.observation_model.t()) + &self.measurement_noise_covariance
    }

    // Chi-Square test for hypothesis testing
    fn chi_square_test(&self, observation: &Array2<f64>, t: usize) -> f64 {
        let innovation = self.innovation(observation, t);
        let innovation_cov = self.innovation_covariance(t);
        let inv_innovation_cov = innovation_cov.inv().unwrap();
        innovation.t().dot(&inv_innovation_cov).dot(&innovation)[[0, 0]]
    }

    // Method to calculate the Mean Squared Error (MSE)
    fn mean_squared_error(&self, true_state: &Array2<f64>) -> f64 {
        let error = true_state - &self.state_estimate;
        error.mapv(|e| e.powi(2)).mean().unwrap()
    }
}