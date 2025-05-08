cmake_minimum_required(VERSION 3.22.1)
project(AndroidEngine)

add_library(androidengine SHARED
    app/src/main/cpp/AndroidEngine.cpp)

# اضافه کردن مسیر هدرهای ضروری
include_directories(
    ${ANDROID_NDK}/sources/android/native_app_glue
)

find_library(log-lib log)

target_link_libraries(androidengine
    android
    ${log-lib})