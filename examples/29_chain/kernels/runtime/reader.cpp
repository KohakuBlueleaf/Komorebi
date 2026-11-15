// Chain example in the runtime's form, reader: the kernel form's steps, called by the
// runtime's reader for each command (its start zeroes the chain's words).
#include "../reader.cpp"

void km_program(km::Reader& reader, const km_command_t& command) {
    command_of(reader, command);
}
