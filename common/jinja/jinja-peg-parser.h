#pragma once

#include "jinja-vm.h"

#include <string>

namespace jinja {

struct jinja_parse_options {
    bool trim_blocks = false;
    bool lstrip_blocks = false;
};

program parse_with_peg(const std::string & input, const jinja_parse_options & options = {});

} // namespace jinja
