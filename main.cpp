#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <cmath>
// 全局變量
float bodyAngle = 0.0f;       // 身體旋轉角度
float leftUpperArmAngle = 0.0f;   // 左上臂角度
float leftForeArmAngle = 0.0f;    // 左前臂角度
float rightUpperArmAngle = 0.0f;  // 右上臂角度
float rightForeArmAngle = 0.0f;   // 右前臂角度
float cameraRotation = 0.0f;      // 攝像機角度
const float MAX_UPPER_ARM_ANGLE = 45.0f;   // 上臂最大角度
const float MIN_UPPER_ARM_ANGLE = -90.0f;  // 上臂最小角度
const float MAX_FORE_ARM_ANGLE = 120.0f;   // 前臂最大角度
const float MIN_FORE_ARM_ANGLE = -5.0f;    // 前臂最小角度

// 輔助函數：限制角度在指定範圍內
float clampAngle(float angle, float min, float max) {
    if (angle > max) return max;
    if (angle < min) return min;
    return angle;
}
// 頂點著色器
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform vec3 color;    // 添加顏色uniform
    out vec3 fragColor;    // 傳遞給片段著色器
    
    void main() {
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        fragColor = color;  // 傳遞顏色
    }
)";

// 更新片段著色器
const char* fragmentShaderSource = R"(
    #version 330 core
    in vec3 fragColor;
    out vec4 FragColor;
    
    void main() {
        FragColor = vec4(fragColor, 1.0);
    }
)";

GLuint shaderProgram;
GLuint VAO, VBO;

// 初始化著色器
void checkShaderCompileErrors(GLuint shader) {
    int success;
    char infoLog[512];
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
}

void initShaders() {
    // 創建頂點著色器
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    checkShaderCompileErrors(vertexShader);

    // 創建片段著色器
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    checkShaderCompileErrors(fragmentShader);

    // 創建並連接著色器程序
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // 檢查程序連接錯誤
    int success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    // 刪除著色器，因為它們已經連接到程序中
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}
void createCube() {
    float vertices[] = {
        // 位置              // 顏色             // 法線
        // 前面
        -0.5f, -0.5f,  0.5f, 1.0f, 0.0f, 0.0f,  0.0f,  0.0f,  1.0f,
         0.5f, -0.5f,  0.5f, 1.0f, 0.0f, 0.0f,  0.0f,  0.0f,  1.0f,
         0.5f,  0.5f,  0.5f, 1.0f, 0.0f, 0.0f,  0.0f,  0.0f,  1.0f,
         0.5f,  0.5f,  0.5f, 1.0f, 0.0f, 0.0f,  0.0f,  0.0f,  1.0f,
        -0.5f,  0.5f,  0.5f, 1.0f, 0.0f, 0.0f,  0.0f,  0.0f,  1.0f,
        -0.5f, -0.5f,  0.5f, 1.0f, 0.0f, 0.0f,  0.0f,  0.0f,  1.0f,

        // 後面
        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 0.0f,  0.0f,  0.0f, -1.0f,
         0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 0.0f,  0.0f,  0.0f, -1.0f,
         0.5f,  0.5f, -0.5f, 0.0f, 1.0f, 0.0f,  0.0f,  0.0f, -1.0f,
         0.5f,  0.5f, -0.5f, 0.0f, 1.0f, 0.0f,  0.0f,  0.0f, -1.0f,
        -0.5f,  0.5f, -0.5f, 0.0f, 1.0f, 0.0f,  0.0f,  0.0f, -1.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 1.0f, 0.0f,  0.0f,  0.0f, -1.0f,

        // 左面
        -0.5f,  0.5f,  0.5f, 0.0f, 0.0f, 1.0f, -1.0f,  0.0f,  0.0f,
        -0.5f,  0.5f, -0.5f, 0.0f, 0.0f, 1.0f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 1.0f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f, -0.5f, 0.0f, 0.0f, 1.0f, -1.0f,  0.0f,  0.0f,
        -0.5f, -0.5f,  0.5f, 0.0f, 0.0f, 1.0f, -1.0f,  0.0f,  0.0f,
        -0.5f,  0.5f,  0.5f, 0.0f, 0.0f, 1.0f, -1.0f,  0.0f,  0.0f,

        // 右面
         0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 0.0f,  1.0f,  0.0f,  0.0f,
         0.5f,  0.5f, -0.5f, 1.0f, 1.0f, 0.0f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 0.0f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f, -0.5f, 1.0f, 1.0f, 0.0f,  1.0f,  0.0f,  0.0f,
         0.5f, -0.5f,  0.5f, 1.0f, 1.0f, 0.0f,  1.0f,  0.0f,  0.0f,
         0.5f,  0.5f,  0.5f, 1.0f, 1.0f, 0.0f,  1.0f,  0.0f,  0.0f,

        // 底面
        -0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 1.0f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 1.0f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f,  0.5f, 1.0f, 0.0f, 1.0f,  0.0f, -1.0f,  0.0f,
         0.5f, -0.5f,  0.5f, 1.0f, 0.0f, 1.0f,  0.0f, -1.0f,  0.0f,
        -0.5f, -0.5f,  0.5f, 1.0f, 0.0f, 1.0f,  0.0f, -1.0f,  0.0f,
        -0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 1.0f,  0.0f, -1.0f,  0.0f,

        // 上面
        -0.5f,  0.5f, -0.5f, 0.0f, 1.0f, 1.0f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f, -0.5f, 0.0f, 1.0f, 1.0f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f, 0.0f, 1.0f, 1.0f,  0.0f,  1.0f,  0.0f,
         0.5f,  0.5f,  0.5f, 0.0f, 1.0f, 1.0f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f,  0.5f, 0.0f, 1.0f, 1.0f,  0.0f,  1.0f,  0.0f,
        -0.5f,  0.5f, -0.5f, 0.0f, 1.0f, 1.0f,  0.0f,  1.0f,  0.0f
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // 位置屬性
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // 顏色屬性
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // 法線屬性
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    
    float rotationSpeed = 2.0f;  // 可以調整旋轉速度

    // 身體旋轉
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        bodyAngle += rotationSpeed;
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
        bodyAngle -= rotationSpeed;
    
    // 左手臂控制
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        leftUpperArmAngle += rotationSpeed;  // 向後抬
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        leftUpperArmAngle -= rotationSpeed;  // 向前放
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        leftForeArmAngle += rotationSpeed;   // 手肘彎曲
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        leftForeArmAngle -= rotationSpeed;   // 手肘伸直

    // 右手臂控制
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        rightUpperArmAngle += rotationSpeed;  // 向後抬
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        rightUpperArmAngle -= rotationSpeed;  // 向前放
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
        rightForeArmAngle += rotationSpeed;   // 手肘彎曲
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
        rightForeArmAngle -= rotationSpeed;   // 手肘伸直

    // 應用角度限制
    leftUpperArmAngle = clampAngle(leftUpperArmAngle, MIN_UPPER_ARM_ANGLE, MAX_UPPER_ARM_ANGLE);
    leftForeArmAngle = clampAngle(leftForeArmAngle, MIN_FORE_ARM_ANGLE, MAX_FORE_ARM_ANGLE);
    rightUpperArmAngle = clampAngle(rightUpperArmAngle, MIN_UPPER_ARM_ANGLE, MAX_UPPER_ARM_ANGLE);
    rightForeArmAngle = clampAngle(rightForeArmAngle, MIN_FORE_ARM_ANGLE, MAX_FORE_ARM_ANGLE);

    // 視角控制（如果需要）
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        cameraRotation += rotationSpeed;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        cameraRotation -= rotationSpeed;
}

void drawRobot(const glm::mat4& projection, const glm::mat4& view) {
    glUseProgram(shaderProgram);
    
    // 獲取uniform位置
    GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
    GLuint colorLoc = glGetUniformLocation(shaderProgram, "color");

    // 設置view和projection矩陣
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

    // 基礎變換
    glm::mat4 baseTransform = glm::mat4(1.0f);
    baseTransform = glm::translate(baseTransform, glm::vec3(0.0f, 0.0f, 0.0f));
    baseTransform = glm::rotate(baseTransform, glm::radians(bodyAngle), glm::vec3(0.0f, 1.0f, 0.0f));

    // 1. 身體核心部分
    // 上身軀幹（胸甲）
    glm::mat4 model = baseTransform;
    model = glm::translate(model, glm::vec3(0.0f, 0.6f, 0.0f));
    model = glm::scale(model, glm::vec3(0.6f, 0.6f, 0.3f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.0f, 0.0f, 0.8f);  // 深藍色胸甲
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 胸部中央裝甲（白色部分）
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.0f, 0.6f, 0.16f));
    model = glm::scale(model, glm::vec3(0.4f, 0.4f, 0.01f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.9f, 0.9f, 0.9f);  // 白色裝甲
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 2. 肩部裝甲
    // 左肩
    model = baseTransform;
    model = glm::translate(model, glm::vec3(-0.4f, 0.8f, 0.0f));
    model = glm::scale(model, glm::vec3(0.3f, 0.25f, 0.25f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 1.0f, 0.5f, 0.0f);  // 橙色肩甲
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 右肩
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.4f, 0.8f, 0.0f));
    model = glm::scale(model, glm::vec3(0.3f, 0.25f, 0.25f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 1.0f, 0.5f, 0.0f);  // 橙色肩甲
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 3. 手臂部分
    // 左上臂
    glm::mat4 leftUpperArmTransform = baseTransform;
    leftUpperArmTransform = glm::translate(leftUpperArmTransform, glm::vec3(-0.4f, 0.6f, 0.0f));
    leftUpperArmTransform = glm::rotate(leftUpperArmTransform, glm::radians(leftUpperArmAngle), glm::vec3(1.0f, 0.0f, 0.0f));
    
    model = leftUpperArmTransform;
    model = glm::scale(model, glm::vec3(0.2f, 0.4f, 0.2f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.7f, 0.7f, 0.7f);  // 銀色上臂
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 左前臂
    glm::mat4 leftForeArmTransform = leftUpperArmTransform;
    leftForeArmTransform = glm::translate(leftForeArmTransform, glm::vec3(0.0f, -0.4f, 0.0f));
    leftForeArmTransform = glm::rotate(leftForeArmTransform, glm::radians(leftForeArmAngle), glm::vec3(-1.0f, 0.0f, 0.0f));
    
    model = leftForeArmTransform;
    model = glm::translate(model, glm::vec3(0.0f, -0.2f, 0.0f));
    model = glm::scale(model, glm::vec3(0.15f, 0.4f, 0.15f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.0f, 0.8f, 0.0f);  // 綠色前臂
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 右上臂
    glm::mat4 rightUpperArmTransform = baseTransform;
    rightUpperArmTransform = glm::translate(rightUpperArmTransform, glm::vec3(0.4f, 0.6f, 0.0f));
    rightUpperArmTransform = glm::rotate(rightUpperArmTransform, glm::radians(rightUpperArmAngle), glm::vec3(1.0f, 0.0f, 0.0f));
    
    model = rightUpperArmTransform;
    model = glm::scale(model, glm::vec3(0.2f, 0.4f, 0.2f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.7f, 0.7f, 0.7f);  // 銀色上臂
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 右前臂
    glm::mat4 rightForeArmTransform = rightUpperArmTransform;
    rightForeArmTransform = glm::translate(rightForeArmTransform, glm::vec3(0.0f, -0.4f, 0.0f));
    rightForeArmTransform = glm::rotate(rightForeArmTransform, glm::radians(rightForeArmAngle), glm::vec3(-1.0f, 0.0f, 0.0f));
    
    model = rightForeArmTransform;
    model = glm::translate(model, glm::vec3(0.0f, -0.2f, 0.0f));
    model = glm::scale(model, glm::vec3(0.15f, 0.4f, 0.15f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.0f, 0.8f, 0.0f);  // 綠色前臂
    glDrawArrays(GL_TRIANGLES, 0, 36);
    }

int main() {
    // 初始化 GLFW
    if (!glfwInit()) {
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "3D Robot Arm", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        return -1;
    }

    // 初始化著色器
    initShaders();

    createCube();
    glEnable(GL_DEPTH_TEST);

    // 渲染循環
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 視圖矩陣
        float radius = 3.0f;  // 增加攝像機距離
        float camX = radius * sin(glm::radians(cameraRotation));
        float camZ = radius * cos(glm::radians(cameraRotation));
        glm::mat4 view = glm::lookAt(
            glm::vec3(camX, 2.0f, camZ),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );

        // 投影矩陣
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f/600.0f, 0.1f, 100.0f);

        // 繪製機器人
        drawRobot(projection, view);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 清理資源
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}
