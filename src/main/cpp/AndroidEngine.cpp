// AndroidEngine.cpp
//
// نمونه‌ای از یک موتور گرافیکی بومی اندرویدی به زبان C++ با استفاده از OpenGL ES 3.0
// به عنوان یک فایل تک‌فایلی (بدون وابستگی به کتابخانه‌های خارجی غیر از NDK و android_native_app_glue)

// -----------------------------------------------------------------------------
//  بخش‌های مربوط به سربرگ‌ها و ماکروها
// -----------------------------------------------------------------------------
#include <android_native_app_glue.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "AndroidEngine", __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, "AndroidEngine", __VA_ARGS__))

// -----------------------------------------------------------------------------
//  ساختار حالت موتور (Engine)
// -----------------------------------------------------------------------------
struct Engine {
    ANativeWindow* window;
    EGLDisplay display;
    EGLSurface surface;
    EGLContext context;
    int32_t width;
    int32_t height;

    // اشیای OpenGL
    GLuint program;
    GLuint vao;
    GLuint vbo;

    float angle; // زاویه چرخش مدل (برای انیمیشن)
};

static Engine engine = {0};

// -----------------------------------------------------------------------------
//  توابع ریاضی (ماتریس‌ها)
// -----------------------------------------------------------------------------
static void mat4_identity(float* m) {
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void mat4_mul(float* result, const float* a, const float* b) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result[i * 4 + j] =
                a[i * 4 + 0] * b[0 * 4 + j] +
                a[i * 4 + 1] * b[1 * 4 + j] +
                a[i * 4 + 2] * b[2 * 4 + j] +
                a[i * 4 + 3] * b[3 * 4 + j];
        }
    }
}

static void mat4_perspective(float* m, float fovy, float aspect, float near, float far) {
    float f = 1.0f / tanf((fovy * 0.5f) * (3.14159265f / 180.0f));
    memset(m, 0, 16 * sizeof(float));
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = (far + near) / (near - far);
    m[11] = -1.0f;
    m[14] = (2.0f * far * near) / (near - far);
}

static void mat4_lookAt(float* m,
                        float eyeX, floatZ,
                        float centerX, float centerY, float centerZ,
                        float upX, float upY, float upZ)
{
    float f[3] = {centerX - eyeX, centerY - eyeY, centerZ - eyeZ};
    float f_len = sqrtf(f[0]*f[0] + f[1]*f[1] + f[2]*f[2]);
    f[0] /= f_len; f[1] /= f_len; f[2] /= f_len;

    float up[3] = {upX, upY, upZ};
    float up_len = sqrtf(up[0]*up[0] + up[1]*up[1] + up[2]*up[2]);
    up[0] /= up_len; up[1] /= up_len; up[2] /= up_len;

    // s = f x up (cross product)
    float s[3] = {
        f[1]*up[2] - f[2]*up[1],
        f[2]*up[0] - f[0]*up[2],
        f[0]*up[1] - f[1]*up[0]
    };
    float s_len = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    s[0] /= s_len; s[1] /= s_len; s[2] /= s_len;

    // u = s x f
    float u[3] = {
        s[1]*f[2] - s[2]*f[1],
        s[2]*f[0] - s[0]*f[2],
        s[0]*f[1] - s[1]*f[0]
    };

    mat4_identity(m);
    m[0] = s[0];
    m[1] = u[0];
    m[2] = -f[0];
    m[4] = s[1];
    m[5] = u[1];
    m[6] = -f[1];
    m[8] = s[2];
    m[9] = u[2];
    m[10] = -f[2];
    m[12] = - (s[0]*eyeX + s[1]*eyeY + s[2]*eyeZ);
    m[13] = - (u[0]*eyeX + u[1]*eyeY + u[2]*eyeZ);
    m[14] =   f[0]*eyeX + f[1]*eyeY + f[2]*eyeZ;  // در برخی پیاده‌سازی‌ها علامت ممکن است تغییر کند.
}

static void mat4_rotateY(float* m, float angleDegrees) {
    float angle = angleDegrees * 3.14159265f / 180.0f;
    mat4_identity(m);
    m[0] = cosf(angle);
    m[2] = sinf(angle);
    m[8] = -sinf(angle);
    m[10] = cosf(angle);
}

// -----------------------------------------------------------------------------
//  توابع مربوط به شیدرها
// -----------------------------------------------------------------------------
static GLuint loadShader(GLenum type, const char* shaderSrc) {
    GLuint shader = glCreateShader(type);
    if (shader == 0) {
        LOGE("خطا در ایجاد شیدر");
        return 0;
    }
    glShaderSource(shader, 1, &shaderSrc, NULL);
    glCompileShader(shader);
    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint infoLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen) {
            char* buf = (char*) malloc(infoLen);
            if (buf) {
                glGetShaderInfoLog(shader, infoLen, NULL, buf);
                LOGE("خطای کامپایل شیدر %d:\n%s", type, buf);
                free(buf);
            }
        }
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint createProgram(const char* vertSrc, const char* fragSrc) {
    GLuint vertexShader = loadShader(GL_VERTEX_SHADER, vertSrc);
    GLuint fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragSrc);
    GLuint program = glCreateProgram();
    if (program == 0) {
        LOGE("خطا در ایجاد برنامه شیدری");
        return 0;
    }
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint infoLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen) {
            char* buf = (char*) malloc(infoLen);
            if (buf) {
                glGetProgramInfoLog(program, infoLen, NULL, buf);
                LOGE("خطا در لینک کردن برنامه شیدری:\n%s", buf);
                free(buf);
            }
        }
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

// شیدرهای مدرن OpenGL ES 3.0 (با GLSL ES 300)
static const char* vertexShaderSource = R"(
    #version 300 es
    layout(location = 0) in vec3 aPos;
    layout(location = 1) in vec3 aNormal;

    uniform mat4 uModel;
    uniform mat4 uView;
    uniform mat4 uProjection;

    out vec3 FragPos;
    out vec3 Normal;

    void main() {
        FragPos = vec3(uModel * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(uModel))) * aNormal;
        gl_Position = uProjection * uView * vec4(FragPos, 1.0);
    }
)";

static const char* fragmentShaderSource = R"(
    #version 300 es
    precision mediump float;

    in vec3 FragPos;
    in vec3 Normal;

    uniform vec3 uLightPos;
    uniform vec3 uViewPos;
    uniform vec3 uLightColor;
    uniform vec3 uObjectColor;

    out vec4 FragColor;

    void main(){
        // مؤلفه محیطی
        float ambientStrength = 0.1;
        vec3 ambient = ambientStrength * uLightColor;
        
        // مؤلفه پراکنده (diffuse)
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(uLightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * uLightColor;
        
        // مؤلفه آینه‌ای (specular)
        float specularStrength = 0.5;
        vec3 viewDir = normalize(uViewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 specular = specularStrength * spec * uLightColor;
        
        vec3 result = (ambient + diffuse + specular) * uObjectColor;
        FragColor = vec4(result, 1.0);
    }
)";

// -----------------------------------------------------------------------------
//  تابع راه‌اندازی EGL و ایجاد کانتکست OpenGL ES 3.0
// -----------------------------------------------------------------------------
static bool initEGL(Engine* eng, ANativeWindow* win) {
    eng->window = win;
    const EGLint configAttribs[] = {
         EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
         EGL_SURFACE_TYPE,    EGL_WINDOW_BIT,
         EGL_RED_SIZE,        8,
         EGL_GREEN_SIZE,      8,
         EGL_BLUE_SIZE,       8,
         EGL_ALPHA_SIZE,      8,
         EGL_DEPTH_SIZE,      24,
         EGL_STENCIL_SIZE,    8,
         EGL_NONE
    };
    EGLint major, minor, numConfigs;
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
       LOGE("غیرقابل دسترسی بودن نمایش EGL");
       return false;
    }
    if (!eglInitialize(display, &major, &minor)) {
       LOGE("عدم توانایی در مقداردهی اولیه EGL");
       return false;
    }
    EGLConfig config;
    if (!eglChooseConfig(display, configAttribs, &config, 1, &numConfigs)) {
       LOGE("عدم توانایی در انتخاب پیکربندی EGL");
       return false;
    }
    EGLSurface surface = eglCreateWindowSurface(display, config, win, NULL);
    if (surface == EGL_NO_SURFACE) {
       LOGE("عدم توانایی در ایجاد سطح پنجره EGL");
       return false;
    }
    const EGLint contextAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
    if (context == EGL_NO_CONTEXT) {
       LOGE("عدم توانایی در ایجاد کانتکست EGL");
       return false;
    }
    if (!eglMakeCurrent(display, surface, surface, context)) {
       LOGE("عدم توانایی در فعال‌سازی کانتکست EGL");
       return false;
    }
    eng->display = display;
    eng->surface = surface;
    eng->context = context;
    
    eglQuerySurface(display, surface, EGL_WIDTH, &eng->width);
    eglQuerySurface(display, surface, EGL_HEIGHT, &eng->height);
    return true;
}

// -----------------------------------------------------------------------------
//  راه‌اندازی OpenGL ES: کامپایل شیدرها، ایجاد VAO/VBO و بارگذاری داده‌های هندسی
// -----------------------------------------------------------------------------
static void initGL(Engine* eng) {
    glViewport(0, 0, eng->width, eng->height);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);

    eng->program = createProgram(vertexShaderSource, fragmentShaderSource);
    if (eng->program == 0) {
        LOGE("خطا در ایجاد برنامه شیدری");
        return;
    }

    // داده‌های مکعب: هر راس شامل 3 مقدار برای موقعیت و 3 برای نرمال است (۶ مختصه در کل)
    static const GLfloat cubeVertices[] = {
        // Front face
        -1.0f, -1.0f,  1.0f,    0.0f,  0.0f,  1.0f,
         1.0f, -1.0f,  1.0f,    0.0f,  0.0f,  1.0f,
         1.0f,  1.0f,  1.0f,    0.0f,  0.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,    0.0f,  0.0f,  1.0f,
         1.0f,  1.0f,  1.0f,    0.0f,  0.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,    0.0f,  0.0f,  1.0f,
        // Back face
        -1.0f, -1.0f, -1.0f,    0.0f,  0.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,    0.0f,  0.0f, -1.0f,
         1.0f,  1.0f, -1.0f,    0.0f,  0.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,    0.0f,  0.0f, -1.0f,
         1.0f,  1.0f, -1.0f,    0.0f,  0.0f, -1.0f,
         1.0f, -1.0f, -1.0f,    0.0f,  0.0f, -1.0f,
        // Left face
        -1.0f, -1.0f, -1.0f,   -1.0f,  0.0f,  0.0f,
        -1.0f, -1.0f,  1.0f,   -1.0f,  0.0f,  0.0f,
        -1.0f,  1.0f,  1.0f,   -1.0f,  0.0f,  0.0f,
        -1.0f, -1.0f, -1.0f,   -1.0f,  0.0f,  0.0f,
        -1.0f,  1.0f,  1.0f,   -1.0f,  0.0f,  0.0f,
        -1.0f,  1.0f, -1.0f,   -1.0f,  0.0f,  0.0f,
        // Right face
         1.0f, -1.0f, -1.0f,    1.0f,  0.0f,  0.0f,
         1.0f,  1.0f,  1.0f,    1.0f,  0.0f,  0.0f,
         1.0f, -1.0f,  1.0f,    1.0f,  0.0f,  0.0f,
         1.0f, -1.0f, -1.0f,    1.0f,  0.0f,  0.0f,
         1.0f,  1.0f, -1.0f,    1.0f,  0.0f,  0.0f,
         1.0f,  1.0f,  1.0f,    1.0f,  0.0f,  0.0f,
        // Top face
        -1.0f,  1.0f, -1.0f,    0.0f,  1.0f,  0.0f,
        -1.0f,  1.0f,  1.0f,    0.0f,  1.0f,  0.0f,
         1.0f,  1.0f,  1.0f,    0.0f,  1.0f,  0.0f,
        -1.0f,  1.0f, -1.0f,    0.0f,  1.0f,  0.0f,
         1.0f,  1.0f,  1.0f,    0.0f,  1.0f,  0.0f,
         1.0f,  1.0f, -1.0f,    0.0f,  1.0f,  0.0f,
        // Bottom face
        -1.0f, -1.0f, -1.0f,    0.0f, -1.0f,  0.0f,
         1.0f, -1.0f,  1.0f,    0.0f, -1.0f,  0.0f,
        -1.0f, -1.0f,  1.0f,    0.0f, -1.0f,  0.0f,
        -1.0f, -1.0f, -1.0f,    0.0f, -1.0f,  0.0f,
         1.0f, -1.0f, -1.0f,    0.0f, -1.0f,  0.0f,
         1.0f, -1.0f,  1.0f,    0.0f, -1.0f,  0.0f
    };

    glGenVertexArrays(1, &eng->vao);
    glBindVertexArray(eng->vao);

    glGenBuffers(1, &eng->vbo);
    glBindBuffer(GL_ARRAY_BUFFER, eng->vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cubeVertices), cubeVertices, GL_STATIC_DRAW);

    // تنظیم attribute موقعیت: location 0، ۳ مقدار (float)، فاصله = 6*sizeof(GLfloat)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);
    // تنظیم attribute نرمال: location 1، شروع از offset برابر با 3*sizeof(GLfloat)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    eng->angle = 0.0f;
}

// -----------------------------------------------------------------------------
//  تابع رندر کردن هر فریم: به‌روزرسانی ماتریس‌ها و رسم اشیای سه‌بعدی
// -----------------------------------------------------------------------------
static void renderFrame(Engine* eng) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // به‌روزرسانی زاویه چرخش
    eng->angle += 1.0f;
    if (eng->angle >= 360.0f)
        eng->angle -= 360.0f;

    // محاسبه ماتریس‌های مدل، دید و پرسپکتیو
    float model[16], view[16], projection[16];
    mat4_identity(model);
    float rotY[16];
    mat4_rotateY(rotY, eng->angle);
    mat4_mul(model, model, rotY);

    // ایجاد ماتریس دید (دوربین در (0,0,5) به سمت مبدا)
    mat4_lookAt(view, 0.0f, 0.0f, 5.0f,
                    0.0f, 0.0f, 0.0f,
                    0.0f, 1.0f, 0.0f);
    // ماتریس پرسپکتیو (فیلد دید 60 درجه)
    float aspect = (float)eng->width / (float)eng->height;
    mat4_perspective(projection, 60.0f, aspect, 0.1f, 100.0f);

    // استفاده از برنامه شیدری
    glUseProgram(eng->program);

    // ارسال ماتریس‌ها به شیدر (بدست آوردن مکان uniformها)
    GLint locModel = glGetUniformLocation(eng->program, "uModel");
    GLint locView  = glGetUniformLocation(eng->program, "uView");
    GLint locProj  = glGetUniformLocation(eng->program, "uProjection");
    glUniformMatrix4fv(locModel, 1, GL_FALSE, model);
    glUniformMatrix4fv(locView, 1, GL_FALSE, view);
    glUniformMatrix4fv(locProj, 1, GL_FALSE, projection);

    // تنظیم پارامترهای نورپردازی
    GLint locLightPos   = glGetUniformLocation(eng->program, "uLightPos");
    GLint locViewPos    = glGetUniformLocation(eng->program, "uViewPos");
    GLint locLightColor = glGetUniformLocation(eng->program, "uLightColor");
    GLint locObjColor   = glGetUniformLocation(eng->program, "uObjectColor");
    float lightPos[3]   = { 5.0f, 5.0f, 5.0f };
    float viewPos[3]    = { 0.0f, 0.0f, 5.0f };
    float lightColor[3] = { 1.0f, 1.0f, 1.0f };
    float objColor[3]   = { 0.8f, 0.5f, 0.2f };
    glUniform3fv(locLightPos, 1, lightPos);
    glUniform3fv(locViewPos, 1, viewPos);
    glUniform3fv(locLightColor, 1, lightColor);
    glUniform3fv(locObjColor, 1, objColor);

    // رسم مکعب
    glBindVertexArray(eng->vao);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);

    eglSwapBuffers(eng->display, eng->surface);
}

// -----------------------------------------------------------------------------
//  آزادسازی منابع EGL و OpenGL
// -----------------------------------------------------------------------------
static void termEGL(Engine* eng) {
    if (eng->display != EGL_NO_DISPLAY) {
        eglMakeCurrent(eng->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (eng->context != EGL_NO_CONTEXT)
            eglDestroyContext(eng->display, eng->context);
        if (eng->surface != EGL_NO_SURFACE)
            eglDestroySurface(eng->display, eng->surface);
        eglTerminate(eng->display);
    }
    eng->display = EGL_NO_DISPLAY;
    eng->context = EGL_NO_CONTEXT;
    eng->surface = EGL_NO_SURFACE;
}

// -----------------------------------------------------------------------------
//  تابع پردازش دستورات سیستم اندروید (مانند ایجاد، خاتمه پنجره)
// -----------------------------------------------------------------------------
static void handleCmd(struct android_app* app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            if (app->window != NULL) {
                if (initEGL(&engine, app->window)) {
                    initGL(&engine);
                    LOGI("EGL و OpenGL مقداردهی اولیه شدند: %dx%d", engine.width, engine.height);
                }
            }
            break;
        case APP_CMD_TERM_WINDOW:
            termEGL(&engine);
            break;
        default:
            break;
    }
}

// -----------------------------------------------------------------------------
//  تابع اصلی بومی اندروید (نقطه ورود برنامه)
// -----------------------------------------------------------------------------
void android_main(struct android_app* app) {
    app->onAppCmd = handleCmd;
    engine.angle = 0.0f;

    // حلقه اصلی: پردازش رویدادهای سیستم و مناسب‌سازی فریم‌ها
    while (1) {
        int events;
        struct android_poll_source* source;
        while (ALooper_pollAll(0, NULL, &events, (void**)&source) >= 0) {
            if (source != NULL) {
                source->process(app, source);
            }
            if (app->destroyRequested != 0) {
                termEGL(&engine);
                return;
            }
        }
        if (engine.display != EGL_NO_DISPLAY)
            renderFrame(&engine);
    }
}
