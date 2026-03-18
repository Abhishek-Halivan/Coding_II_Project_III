#include "FaceDetectionApp.h"
#include <iostream>

int main(int argc, char** argv) {
    std::string cascadePath;

    if (argc > 1) {
        cascadePath = argv[1];
    }

    FaceDetectionApp app;

    if (!app.initialize(cascadePath)) {
        return 1;
    }

    app.run();

    return 0;
}
