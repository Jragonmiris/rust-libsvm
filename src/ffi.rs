extern crate libc;

use self::libc::{c_int, c_double, c_char};
use ::SvmNode;
use std::default::Default;

#[repr(C)]
pub struct CSvmProblem {
    pub l: i32,
    pub y: *mut f64,
    pub x: *mut *mut SvmNode
}

#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum SvmType {
    CSvc,
    NuSvc,
    OneClass,
    EpsilonSvr,
    NuSvr,
}

#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum KernelType {
    Linear,
    Poly,
    Rbf,
    Sigmoid,
    Precomputed,
}

#[repr(C)]
pub struct CSvmParameter {
    pub svm_type: SvmType,
    pub kernel_type: KernelType,
    pub degree: i32,
    pub gamma: f64,
    pub coef0: f64,

    pub cache_size: f64,
    pub eps: f64,
    pub c: f64,
    pub nr_weight: i32,
    pub weight_label: *mut i32,
    pub weight: *mut f64,
    pub nu: f64,
    pub p: f64,
    pub shrinking: bool,
    pub probability: bool,
}

impl Default for CSvmParameter {
  fn default() -> CSvmParameter {
    use std::ptr;
    CSvmParameter {
      svm_type: SvmType::CSvc,
      kernel_type: KernelType::Linear,
      degree: 0,
      gamma: 0.0,
      coef0: 0.0,

      cache_size: 0.0,
      eps: 0.0,
      c: 0.0,
      nr_weight: 0,
      weight_label: ptr::null_mut(),
      weight: ptr::null_mut(),
      nu: 0.0,
      p: 0.0,
      shrinking: false,
      probability: false,
    }
  }
}

#[repr(C)]
pub struct CSvmModel {
    pub param: CSvmParameter,
    pub nr_class: i32,
    pub l: i32,
    pub sv: *mut *mut SvmNode,
    pub sv_coef: *mut *mut f64,
    pub rho: *mut f64,
    pub prob_a: *mut f64,
    pub prob_b: *mut f64,
    pub sv_indices: *mut i32,

    pub label: *mut i32,
    pub n_sv: *mut i32,

    pub free_sv: bool,
}

#[link(name = "svm")]
#[allow(dead_code)]
extern "C" {
    pub fn svm_train(prob: *const CSvmProblem, param: *const CSvmParameter) -> *mut CSvmModel;
    pub fn svm_cross_validation(svm_problem: *const CSvmProblem,
                            param: *const CSvmParameter,
                            nr_fold: c_int,
                            target: *mut c_double);

    pub fn svm_save_model(model_file_name: *const c_char, model: *const CSvmModel) -> c_int;
    pub fn svm_load_model(model_file_name: *const c_char) -> *mut CSvmModel;

    pub fn svm_get_svm_type(model: *const CSvmModel) -> c_int;
    pub fn svm_get_nr_class(model: *const CSvmModel) -> c_int;
    pub fn svm_get_labels(model: *const CSvmModel, label: *mut c_int);
    pub fn svm_get_sv_indices(model: *const CSvmModel, sv_indices: *mut c_int);
    pub fn svm_get_nr_sv(model: *const CSvmModel) -> c_int;
    pub fn svm_get_svr_probability(mode: *const CSvmModel) -> c_double;

    pub fn svm_predict_values(model: *const CSvmModel,
                          x: *const SvmNode,
                          dec_values: *mut c_double)
                          -> c_double;
    pub fn svm_predict(model: *const CSvmModel, x: *const SvmNode) -> c_double;
    pub fn svm_predict_probability(model: *const CSvmModel,
                               x: *const SvmNode,
                               prob_estimates: *mut c_double)
                               -> c_double;

    pub fn svm_free_model_content(model_ptr: *mut CSvmModel);
    pub fn svm_free_and_destroy_model(model_ptr_ptr: *mut *mut CSvmModel);
    pub fn svm_free_param(param: *mut CSvmParameter);

    pub fn svm_check_parameter(prob: *const CSvmProblem, param: *const CSvmParameter) -> *const c_char;
    pub fn svm_check_probability_model(model: *const CSvmModel) -> c_int;

    pub fn svm_set_print_string_function(func: extern fn(*const c_char));
}