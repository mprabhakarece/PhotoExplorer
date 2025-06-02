cmake_minimum_required(VERSION 3.16)
project(PhotoBrowser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)

# --- Qt6 ---
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# --- OpenCV ---
set(OpenCV_DIR "C:/libs/opencv/build")  # ✅ Change this to your actual OpenCV path
find_package(OpenCV REQUIRED)

# --- Dlib ---
add_subdirectory("C:/libs/dlib" dlib_build)  # ✅ Change to actual dlib source path

# --- Include paths ---
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "C:/libs/dlib"                           # ✅ Needed for dlib headers
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- Source files ---
add_executable(PhotoBrowser
    main.cpp
    mainwindow.cpp
    mainwindow.h
    faceindexer.cpp       # (to be added later)
    faceindexer.h
)

# --- Link libraries ---
target_link_libraries(PhotoBrowser
    Qt6::Widgets
    ${OpenCV_LIBS}
    dlib
)

# === Copy Dlib model files to ./models next to executable ===
file(COPY "${CMAKE_SOURCE_DIR}/models/shape_predictor_68_face_landmarks.dat"
     DESTINATION "${CMAKE_BINARY_DIR}/models")

file(COPY "${CMAKE_SOURCE_DIR}/models/dlib_face_recognition_resnet_model_v1.dat"
     DESTINATION "${CMAKE_BINARY_DIR}/models")

