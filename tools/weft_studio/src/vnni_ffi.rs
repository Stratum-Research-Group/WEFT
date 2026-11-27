//! Rust FFI bindings for Native WEFT AVX-512 VNNI Speculative Recurrent Kernels.

use std::os::raw::c_int;

#[link(name = "weft_vnni", kind = "dylib")]
extern "C" {
    pub fn weft_speculative_vnni_step(
        weights: *const u8,
        inputs: *const u8,
        outputs: *mut i32,
        dim: usize,
        tokens: usize,
    ) -> c_int;
}

/// Safe Rust wrapper for AVX-512 VNNI speculative recurrent execution.
pub fn speculative_vnni_eval(weights: &[u8], inputs: &[u8], dim: usize, tokens: usize) -> Result<Vec<i32>, &'static str> {
    if weights.len() < dim * dim || inputs.len() < tokens * dim || dim % 64 != 0 {
        return Err("Invalid buffer lengths or dimension not multiple of 64");
    }

    let mut outputs = vec![0i32; tokens * dim];
    let ret = unsafe {
        weft_speculative_vnni_step(
            weights.as_ptr(),
            inputs.as_ptr(),
            outputs.as_mut_ptr(),
            dim,
            tokens,
        )
    };

    if ret == 0 {
        Ok(outputs)
    } else {
        Err("AVX-512 kernel execution error")
    }
}
