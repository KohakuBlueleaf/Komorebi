// Chain example in the runtime's form, compute: the kernel form's, called by the
// runtime's compute for each command.
#include "../compute.cpp"

void km_program(km::Compute& compute, const km_command_t& command) {
    command_of(compute, command);
}
