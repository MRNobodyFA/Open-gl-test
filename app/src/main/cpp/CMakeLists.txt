cmake_minimum_required(VERSION 3.4.1)

# ایجاد کتابخانه اشتراکی "androidengine" از فایل‌های C++ (در این مثال تنها AndroidEngine.cpp)
add_library(androidengine SHARED
            AndroidEngine.cpp)

# پیدا کردن کتابخانه log (برای خروجی log در اندروید)
find_library(log-lib
             log)

# لینک کردن کتابخانه‌های مورد نیاز: EGL و GLESv3
target_link_libraries(androidengine ${log-lib} EGL GLESv3)
