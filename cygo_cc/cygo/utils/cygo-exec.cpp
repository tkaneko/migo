#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/module/module.h>
#include <iostream>

using namespace ::executorch::extension;

void usage(const char *name) {
    std::cout << name << " path/to/model.pte" << "\n";
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        usage(argv[0]);
        return 1;
    }
    Module module("/path/to/model.pte");
}
