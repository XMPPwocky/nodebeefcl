uint run_rng(uint state, uint a) {
    return (a * (state & 0xFFFF)) + (state >> 16);
}

__kernel void node_newer_rng(
    const uint search_len,
    const uint successful_guesses_max,
    const uint a,

    __global const uint *outputs,
    const uint outputs_len,
    __global uint *successful_guesses_count,
    __global uint *successful_guesses
) {
    uint gid = get_global_id(0);
    uint guess = gid;

    uint state = outputs[0] | (guess<<16);

    uint cur_output = 1;
    for (uint i = 0; i < search_len && cur_output < outputs_len; i++) {
        *&state = run_rng(state, a);
        if ((state&0xFFFF) == outputs[cur_output]) {
            cur_output += 1;
            i = 0;
        }
    }

    if (cur_output == outputs_len) {
        uint guess_idx = atomic_inc(successful_guesses_count);
        if (guess_idx < successful_guesses_max) {
            successful_guesses[guess_idx] = state;
        }
    }
}
