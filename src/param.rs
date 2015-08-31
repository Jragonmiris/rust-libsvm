use ::ffi::{CSvmParameter, KernelType, SvmType};
use std::default::Default;
use std::cell::RefCell;

/// The parameters needed for certain Kernel types.
#[derive(Debug,Clone,Copy)]
pub enum KernelParam {
	Linear,
    Poly{degree: i32, gamma: f64, coef0: f64},
    Rbf{gamma: f64},
    Sigmoid{gamma: f64, coef0: f64},
    Precomputed,
}

impl KernelParam {
	pub fn to_kernel_type(&self) -> KernelType {
		use KernelParam::*;
		match *self {
			Linear => KernelType::Linear,
			Poly{..} => KernelType::Poly,
			Rbf{..} => KernelType::Rbf,
			Sigmoid{..} => KernelType::Sigmoid,
			Precomputed => KernelType::Precomputed,
		}
	}
}

/// This is a representation of the weights used for CSVC in libsvm.
/// It enforces one label per one weight.
#[derive(Debug,Clone)]
pub struct Weight{pub label: i32, pub weight: f64}

/// The parameters needed for certain SVM types.
/// Note that unlike the C library where the weights in CSVC
/// are split into nr_weights, weights, and weight_labels, this is
/// all encoded into a single vector to ensure the lengths match. This will
/// be converted into the correct lists internally.
#[derive(Debug,Clone)]
pub enum SvmTypeParam {
	CSvc{c: f64, weights: Vec<Weight>},
    NuSvc{nu: f64},
    OneClass{nu: f64},
    EpsilonSvr{p: f64},
    NuSvr{nu: f64},
}

impl SvmTypeParam {
	pub fn to_svm_type(&self) -> SvmType {
		use SvmTypeParam::*;
		match *self {
			CSvc{..} => SvmType::CSvc,
			NuSvc{..} => SvmType::NuSvc,
			OneClass{..} => SvmType::OneClass,
			EpsilonSvr{..} => SvmType::EpsilonSvr,
			NuSvr{..} => SvmType::NuSvr,
		}
	}
}

#[derive(Clone,Debug)]
/// This is a set of parameters for generating a model. It is a Rust representation of the
/// C struct svm_parameter, and can be converted into a C struct internally. It is built to be more
/// "Rustic". The C version has many unimportant and unread fields if certain kernel or parameter
// types are used, whereas this one only requires you to set the fields necessary
/// using the KernelParam and SvmTypeParam fields.
///
/// This will be correctly converted into the C representation, and any potentially shared memory
/// is handled by this library in concert with libsvm. It should not be possible to invalidate
/// anything for force a read/write after free from libsvm with this.
///
/// For more information on specific parameters, see the liibsvm documentation.
pub struct SvmParameter {
	pub kernel_param: KernelParam,
	pub svm_type_param: SvmTypeParam,
	pub shrinking: bool,
	pub probability: bool,
	pub cache_size: f64,
	pub epsilon: f64,

	// This may be a bit confusing. According to the libsvm documentation,
	// memory from svm_parameter may be referenced by an svm_model. So what we do
	// is construct these vectors when making the C representation, then cache them.
	//
	// You may notice that in the crep function, we invalidate the cache every time. That's because
	// svm_type_param is public, so if someone queries the c-library twice, changing the type param,
	// we need it to recalculate the weight vectors.
	//
	// So why can we do this? Well, the only time the persistence of these vectors matters is after
	// this parameter is used by svm_train. In our library, we define this as MOVING the SvmParameter
	// into the newly-made SvmModel, so after svm_train is called, and the only way to view them again involves a clone.
	// The parameters can never be modified again, and so we "turn off" the recomputation of 
	// the vectors so we don't invalidate any memory.
	weight_labels: RefCell<Option<Vec<i32>>>,
	weights: RefCell<Option<Vec<f64>>>,
	in_model: bool,
}

impl SvmParameter {
	/// Builds a new SvmParameter struct from all the necessary fields.
	pub fn new(kernel_param: KernelParam, svm_type_param: SvmTypeParam, shrinking: bool, probability: bool,
		cache_size: f64, epsilon: f64) -> SvmParameter {
		SvmParameter {
			kernel_param: kernel_param,
			svm_type_param: svm_type_param,
			shrinking: shrinking,
			probability: probability,
			cache_size: cache_size,
			epsilon: epsilon,
			weight_labels: RefCell::new(None),
			weights: RefCell::new(None),
			in_model: false,
		}
	}

	fn from_crep(crep: &CSvmParameter) -> SvmParameter {
		use KernelType::*;
		use SvmType::*;

		SvmParameter {
			in_model: false,
			weight_labels: RefCell::new(None),
			weights: RefCell::new(None),

			shrinking: crep.shrinking,
			probability: crep.probability,
			cache_size: crep.cache_size,
			epsilon: crep.eps,

			kernel_param: match crep.kernel_type {
				Linear => KernelParam::Linear,
				Poly => KernelParam::Poly{degree: crep.degree, gamma: crep.gamma, coef0: crep.coef0},
				Rbf => KernelParam::Rbf{gamma: crep.gamma},
				Sigmoid => KernelParam::Sigmoid{gamma: crep.gamma, coef0: crep.coef0},
				Precomputed => KernelParam::Precomputed,
			},

			svm_type_param: match crep.svm_type {
				CSvc => SvmTypeParam::CSvc{
					c: crep.c, 
					weights: make_weights(crep.nr_weight, crep.weight_label, crep.weight)
				},
				NuSvc => SvmTypeParam::NuSvc{nu: crep.nu},
				OneClass => SvmTypeParam::OneClass{nu: crep.nu},
				EpsilonSvr => SvmTypeParam::EpsilonSvr{p: crep.p},
				NuSvr => SvmTypeParam::NuSvr{nu: crep.nu},
			}
		}
	}

	fn crep(&self) -> CSvmParameter {
		use SvmTypeParam::*;
		use KernelParam::*;

		if !self.in_model {
			self.invalidate_cache();
		}

		let mut c_params: CSvmParameter = CSvmParameter::default();

		c_params.kernel_type = self.kernel_param.to_kernel_type();
		match self.kernel_param {
			Poly{degree, gamma, coef0} => {
				c_params.degree = degree;
				c_params.gamma = gamma;
				c_params.coef0 = coef0;
			},
			Rbf{gamma} => {
				c_params.gamma = gamma;
			},
			Sigmoid{gamma, coef0} => {
				c_params.gamma = gamma;
				c_params.coef0 = coef0;
			},
			Linear => {},
			Precomputed => {},
		};

		c_params.svm_type = self.svm_type_param.to_svm_type();
		match self.svm_type_param {
			CSvc{c, ref weights} => {
				c_params.c = c;

				self.cache_weights(weights);
				c_params.nr_weight = weights.len() as i32;
				c_params.weight = self.weights.borrow_mut().as_mut().unwrap().as_mut_ptr();
				c_params.weight_label = self.weight_labels.borrow_mut().as_mut().unwrap().as_mut_ptr();
			}
			NuSvc{nu} => {c_params.nu = nu},
			OneClass{nu} => {c_params.nu = nu},
			EpsilonSvr{p} => {c_params.p = p},
			NuSvr{nu} => {c_params.nu = nu},
		};

		c_params.shrinking = self.shrinking;
		c_params.probability = self.probability;
		c_params.cache_size = self.cache_size;
		c_params.eps = self.epsilon;

		c_params
	}

	fn invalidate_cache(&self) {
		*self.weight_labels.borrow_mut() = None;
		*self.weight_labels.borrow_mut() = None;
	}

	fn cache_weights(&self, weights: &Vec<Weight>) {
		if self.weight_labels.borrow().is_some() {
			return;
		}

		let mut raw_weight_labels = Vec::with_capacity(weights.len());
		let mut raw_weights = Vec::with_capacity(weights.len());

		for &Weight{label,weight} in weights.iter() {
			raw_weight_labels.push(label);
			raw_weights.push(weight);
		}

		*self.weight_labels.borrow_mut() = Some(raw_weight_labels);
		*self.weights.borrow_mut() = Some(raw_weights);
	}
}

pub mod protected {
	use super::SvmParameter;
	use ::ffi::CSvmParameter;

	pub fn set_in_model(param: &mut SvmParameter, val: bool) {
		param.in_model = val;
	}

	pub fn param_from_crep(crep: &CSvmParameter) -> SvmParameter {
		SvmParameter::from_crep(crep)
	}

	pub fn crep(param: &SvmParameter) -> CSvmParameter {
		param.crep()
	}
}

fn make_weights(nr_weight: i32, weight_label: *mut i32, weight: *mut f64) -> Vec<Weight> {
	use std::slice;

	if nr_weight == 0 || weight_label.is_null() || weight.is_null() {
		Vec::new()
	} else {
		unsafe {
			let label_slice = slice::from_raw_parts(weight_label, nr_weight as usize);
			let weight_slice = slice::from_raw_parts(weight, nr_weight as usize);

			label_slice.iter().zip(weight_slice.iter())
				.map(|(&label, &weight)| Weight{label:label, weight:weight}).collect()
		}
	}
}