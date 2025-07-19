// Get the mask for a specific index in a 32-bit integer.
//
// You can get the mask with this function then set with `atomicOr`
// or clear with `~` (compliment) then `atomicAnd`.
fn bit_flag(index: u32) -> u32 {
    let word_index = index / 32u;
    let bit_index = index % 32u;
    return 1u << bit_index;
}