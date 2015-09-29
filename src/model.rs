use ::ffi;
use ::ffi::{CSvmModel,SvmType};
use ::param::SvmParameter;
use ::prob::SvmProblem;
use ::datavec::DataVec;
use std::ffi::{CString};
use std::mem;
use std::ops::Drop;
use std::io::{Read,Write};
use std::fs;
use std::fs::File;

use ::rustc_serialize::{Encodable,Decodable,Encoder,Decoder};
use ::tempfile::NamedTempFile;

/// An SVM Model is a trained Support Vector Machine, which can be used
/// to query new problems. It manages all lifetimes and memory needed by itself in
/// concert with libsvm itself (though it may be a little conservative).
// The dead code is actually for lifetimes
#[allow(dead_code)]
pub struct SvmModel<'a> {
	crep: &'a mut CSvmModel,

    param: Option<SvmParameter>,
    prob: Option<SvmProblem>,
}

impl<'a> SvmModel<'a> {
    /// Attempts to save the model to a file and reports whether or
    /// not it was successful. Unfortunately, libsvm doesn't report an
    /// error message so neither can we.
    pub fn save(&self, model_file_name: &str) -> bool {
        unsafe {
            let fname = CString::new(model_file_name).unwrap();

            // returns 0 on success or -1 on failure
            ffi::svm_save_model(fname.as_ptr(), self.crep) == 0
        }
    }

    /// Loads a model from a file.
    pub fn load(model_file_name: &str) -> Self {
        unsafe {
            let fname = CString::new(model_file_name).unwrap();

            SvmModel {
                crep: &mut (*ffi::svm_load_model(fname.as_ptr())),
                param: None,
                prob: None,
            }
        }
    }

    /// Returns the type of the SVM, this is one of the values
    /// of the enum SvmType. Please see the libsvm documentation
    /// for more info.
    pub fn get_svm_type(&self) -> SvmType {
        unsafe {
            mem::transmute(ffi::svm_get_svm_type(self.crep))
        }
    }

    /// Gets the number of possible classes that can be used to label
    /// an input.
    pub fn get_nr_class(&self) -> i32 {
        unsafe {
            ffi::svm_get_nr_class(self.crep) as i32
        }
    }

    /// Get a list of labels that can be used to label an input.
    /// If buf is supplied it will be used as the target and returned
    /// back to you, however, make sure that it is at least get_nr_class in length,
    /// and capacity or bugs could happen.
    pub fn get_labels(&self, buf: Option<Vec<i32>>) -> Vec<i32> {
        let mut buf = match buf {
            None => {
                let size = self.get_nr_class() as usize;

                let mut buf = Vec::with_capacity(size);
                unsafe {
                    buf.set_len(size);
                }

                buf
            },
            Some(buf) => buf,
        };

        unsafe {
            ffi::svm_get_labels(self.crep, buf.as_mut_ptr())
        }

        buf
    }

    /// Returns the number of support vectors.
    pub fn get_nr_sv(&self) -> i32 {
        unsafe {
            ffi::svm_get_nr_sv(self.crep)
        }
    }

    /// Returns the indices of the support vectors.
    pub fn get_sv_indices(&self, buf: Option<Vec<i32>>) -> Vec<i32> {
        let mut buf = match buf {
            None => {
                let size = self.get_nr_sv() as usize;

                let mut buf = Vec::with_capacity(size);
                unsafe {
                    buf.set_len(size);
                }

                buf
            },
            Some(buf) => buf,
        };
        unsafe {
            ffi::svm_get_sv_indices(self.crep, buf.as_mut_ptr());
        }

        buf
    }

    /// Returns the svr probability. This only works if check_probability_model
    /// returns true. Please see libsvm's documentation for more info.
    pub fn get_svr_probability(&self) -> f64 {
        unsafe {
            ffi::svm_get_svr_probability(self.crep) as f64
        }
    }

    /// Predicts the output labels of some input vector, test_vec.
    /// If this is a decision problem, it outputs an array of "arena" decisions
    /// (i.e. label1 vs label2, then label1 vs label3 etc), and the f64 returned is the overall
    /// predicted class.
    ///
    /// If this is a regression, only dec_values[0] is valid and contains the regression info, and the f64
    /// is the same regression value.
    ///
    /// Please see the libsvm documentation for more info, and make sure the dec_values passed in is of the right size.
    /// If none is supplied, it will always allocate as if this is a decision problem.
    pub fn predict_values(&self,
                          test_vec: &DataVec,
                          dec_values: Option<Vec<f64>>)
                          -> (f64, Vec<f64>) {
        let mut dec_values = match dec_values {
            None => {
                let nr_class = self.get_nr_class();
                let len = (nr_class*(nr_class-1)/2) as usize;

                let mut dec_values = Vec::with_capacity(len);
                unsafe {
                    dec_values.set_len(len);
                }

                dec_values
            },
            Some(dec_values) => dec_values,
        };
        let y;
        unsafe {
            y = ffi::svm_predict_values(self.crep, test_vec.as_ptr(), dec_values.as_mut_ptr());
        }

        (y, dec_values)
    }

    /// Predicts the class or regression value of the test vector test_vec.
    /// This is effectively predict_values without the dec_values component.
    pub fn predict(&self, test_vec: &DataVec) -> f64 {
        unsafe {
            ffi::svm_predict(self.crep, test_vec.as_ptr()) as f64
        }
    }

    /// Predicts the class of the feature vector test_vec based on its probability of belonging to a
    /// certain class. This only works correctly if check_probability_model returns true, please check that
    /// first and see the libsvm documentation for more info.
    pub fn predict_probability(&self,
                               test_vec: &DataVec,
                               prob_estimates: Option<Vec<f64>>)
                               -> (f64, Vec<f64>) {
        let mut prob_estimates = match prob_estimates {
            None => {
                let mut prob_estimates = Vec::with_capacity(test_vec.len());
                unsafe {
                    prob_estimates.set_len(test_vec.len());
                }

                prob_estimates
            },
            Some(prob_estimates) => prob_estimates,
        };

        let p;
        unsafe {
            p = ffi::svm_predict_probability(self.crep, test_vec.as_ptr(),
                                        prob_estimates.as_mut_ptr()) as f64;
        }

        (p, prob_estimates)
    }

    /// Tests whether the model has enough information for probability estimates.
    /// Check this before trying get_svr_probability or predict_probability.
    pub fn check_probability_model(&self) -> bool {
        unsafe {
            ffi::svm_check_probability_model(self.crep) != 0
        }
    }

    /// View the parameters this model was generated from.
    /// If this was generated using svm_train from the Rust side, it will
    /// be a clone of the struct used to generate the model. If not, (i.e. it was loaded
    /// from file) it will be generated from the internal C struct.
    ///
    /// Either way, this struct is safe to modify or generate future models.
    pub fn view_params(&self) -> SvmParameter {
        match self.param {
            None => ::param::protected::param_from_crep(&self.crep.param),
            Some(ref param) => {
                let mut p = param.clone();
                ::param::protected::set_in_model(&mut p, false);
                p
            }
        }
    }
}

/// This encodes by saving it to a named temp file and then reading THAT
/// into a Vec<u8> and encoding it. This is probably a bad idea and you should
/// probably use a raw `save` if at all possible.
impl<'a> Encodable for SvmModel<'a> {
    fn encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        // Get a tmp file path by just creating a temp file and getting its handle,
        // then letting it get deleted.
        let path= {
            let file = match NamedTempFile::new() {
                Err(err) => { panic!(err); },
                Ok(file) => file,
            };


            file.path().to_path_buf()
        };

        if !self.save(path.to_str().expect("Could not get file name of temp file")) {
            panic!("Could not save model to temp file");
        }

        let mut file = File::open(&path).expect("Could not open temp file");
        let mut buf = Vec::new();
        if let Err(err) = file.read_to_end(&mut buf) {
            panic!(err);
        }

        if let Err(err) = fs::remove_file(path) {
            panic!(err);
        }

        buf.encode(s)
    }
}

/// This loads the serialized data and then writes it to a tmp file and
/// tells libsvm to load a model from that file. This is probably a dumb idea
/// and you should probably use a raw `load` from a `save`d file if
/// possible.
impl<'a> Decodable for SvmModel<'a> {
    fn decode<D: Decoder>(d: &mut D) -> Result<Self, D::Error> {
        let buf = match Vec::<u8>::decode(d) {
            Err(err) => { return Err(err); },
            Ok(buf) => buf,
        };

        let mut file = match NamedTempFile::new() {
            Err(err) => { panic!(err); },
            Ok(file) => file,
        };

        if let Err(err) = file.write_all(buf.as_slice()) {
            panic!(err);
        }

        Ok(SvmModel::load(file.path().to_str().expect("Could not get file name of temp file")))
    }
}

impl<'a> Drop for SvmModel<'a> {
    fn drop(&mut self) {
        unsafe {
            let mut crep_ref: *mut CSvmModel = self.crep;
            ffi::svm_free_and_destroy_model(&mut crep_ref);
        }
    }
}

pub fn model_from_c_rep(crep: &mut CSvmModel, prob: SvmProblem, mut param: SvmParameter) -> SvmModel {
    ::param::protected::set_in_model(&mut param, true);

    SvmModel {
        crep: crep,
        param: Some(param),
        prob: Some(prob),
    }
}