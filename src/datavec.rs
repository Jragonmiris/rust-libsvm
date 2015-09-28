use ::SvmNode;

use std::cmp::{Ordering};
use std::ops::{Deref,DerefMut};

/// A DataVec is a sparse representation of a vector (usually a feature vector, but
/// possibly a support vector as well).
#[derive(Clone, Debug)]
pub struct DataVec {
    v: Vec<SvmNode>,

    is_sorted: bool,
}

impl DataVec {
    /// Builds a DataVec from a dense vector. That is, a feature vector such as
    /// [3.0, 0, 0, 9.2]
    ///
    /// If any elements are exactly equal to 0.0, they will be filtered out.
    /// If you want to filter it with a different threshold according to your data,
    /// build the sparse vector yourself and construct the DataVec with from_sparse.
    pub fn from_dense(x: Vec<f64>) -> DataVec {
        let mut v = Vec::new();

        for (i, x) in x.into_iter().enumerate() {
            // We shouldn't be in the business of determining what
            // threshold should be filtered out, so only strict 0.0 reps
            // are filtered.
            if x == 0.0 {
                continue;
            }

            v.push(SvmNode((i + 1) as i32, x));
        }

        v.push(SvmNode(-1, 0.0));

        DataVec { v: v, is_sorted: true }
    }

    /// Builds a DataVec from sparse components. The exact format of this is specified in the libsvm
    /// docs, but effectively it's a set of tuples denoting the non-zero elements of the feature vector, starting
    /// at 1.
    ///
    /// (1,3.4), (4,9.2), (-1,0.0) Would correspond to the vector [3.4, 0.0, 0.0, 9.2, ???]  The ??? being that we don't
    /// know if there are any more elements.
    ///
    /// The libsvm documents specify that this must be terminated by a tuple with -1 for the index, and the indices (other than -1)
    /// must be in ascending order. This function takes care of the terminal tuple and sorting for you (but it will not break if either criterion
    /// is met beforehand).
    ///
    /// Malformed input (indices that are lower than 1, but not -1) will panic.
    pub fn from_sparse(mut x: Vec<SvmNode>) -> DataVec {
        DataVec::sort(&mut x);
		DataVec {
		    v: x,
            is_sorted: true,
		}
	}

    /// Sorts the vector again. If the DataVec is ever modified (e.g. via DerefMut),
    /// this sorts it correctly again. This is automatically called by the SvmProb
    /// constructor you usually shouldn't need to worry about this.
    pub fn resort(&mut self) {
        if !self.is_sorted {
            DataVec::sort(self);
            self.is_sorted = true;
        }
    }

    fn sort(x: &mut Vec<SvmNode>) {
        // Sort by the index as in the libsvm docs
        x.sort_by(|a, b| {
            let (&SvmNode(idx1, _), &SvmNode(idx2, _)) = (a, b);

            match (idx1, idx2) {
                (-1, -1) => Ordering::Equal,
                (-1, _) => Ordering::Greater,
                (_, -1) => Ordering::Less,
                (x, y) if x < 1 || y < 1
                    => { panic!("Index is less than 1 but not -1. a: {:?}, b: {:?}", a, b) },
                (x, y) => x.cmp(&y),
            }
        });
        let SvmNode(idx, _) = x[x.len() - 1];

        if idx != -1 {
            x.push(SvmNode(-1, 0.0));
        }
    }
}

impl Deref for DataVec {
    type Target = Vec<SvmNode>;

    fn deref(& self) -> &Self::Target {
        &self.v
    }
}

impl DerefMut for DataVec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.is_sorted = false;
        &mut self.v
    }
}