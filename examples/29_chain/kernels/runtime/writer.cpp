// Chain example in the runtime's form, writer: the kernel form's steps, called by the
// runtime's writer for each command; false: no completion (the command gave up).
#include "../writer.cpp"

bool km_program(km::Writer& writer, const km_command_t& command) {
    return command_of(writer, command);
}
