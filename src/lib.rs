mod datavec;
mod prob;
mod ffi; 
mod model;
mod param;

pub use self::datavec::{DataVec};
pub use self::prob::{SvmProblem};
pub use self::ffi::{KernelType,SvmType,svm_set_print_string_function};
pub use self::model::{SvmModel};
pub use self::param::{SvmParameter,KernelParam,SvmTypeParam};

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct SvmNode(pub i32, pub f64);

pub fn squelch_output() {
	unsafe {
		svm_set_print_string_function(ffi::no_output);
	}
}

mod test {

    #[test]
    fn make_it_link() {

    }
}