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
const float MAX_UPPER_ARM_ANGLE = 90.0f;  // 上臂最大角度
const float MIN_UPPER_ARM_ANGLE = -45.0f;  // 上臂最小角度
const float MAX_FORE_ARM_ANGLE = 0.0f;   // 前臂最大角度
const float MIN_FORE_ARM_ANGLE = -145.0f;  // 前臂最小角度

// 輔助函數：限制角度在指定範圍內
float clampAngle(float angle, float min, float max) {
    if (angle > max) return max;
    if (angle < min) return min;
    return angle;
}
// 頂點著色器
const char *vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aColor;
    layout (location = 2) in vec3 aNormal;
    
    out vec3 ourColor;
    out vec3 Normal;
    out vec3 FragPos;
    
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        gl_Position = projection * view * vec4(FragPos, 1.0);
        ourColor = aColor;
        Normal = mat3(transpose(inverse(model))) * aNormal;
    }
)";

// 片段著色器
const char *fragmentShaderSource = R"(
    #version 330 core
    in vec3 ourColor;
    in vec3 Normal;
    in vec3 FragPos;
    
    out vec4 FragColor;
    
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    
    void main() {
        // 環境光
        float ambientStrength = 0.3;
        vec3 ambient = ambientStrength * ourColor;
        
        // 漫反射
        vec3 lightColor = vec3(1.0, 1.0, 1.0);
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        vec3 result = (ambient + diffuse) * ourColor;
        FragColor = vec4(result, 1.0);
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
    
    // 身體旋轉
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        bodyAngle += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
        bodyAngle -= 1.0f;
    
    // 左臂控制（帶角度限制）
    float oldLeftUpperAngle = leftUpperArmAngle;
    float oldLeftForeAngle = leftForeArmAngle;
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        leftUpperArmAngle += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        leftUpperArmAngle -= 1.0f;
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        leftForeArmAngle += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        leftForeArmAngle -= 1.0f;

    // 限制左臂角度
    leftUpperArmAngle = clampAngle(leftUpperArmAngle, MIN_UPPER_ARM_ANGLE, MAX_UPPER_ARM_ANGLE);
    leftForeArmAngle = clampAngle(leftForeArmAngle, MIN_FORE_ARM_ANGLE, MAX_FORE_ARM_ANGLE);

    // 右臂控制（帶角度限制）
    float oldRightUpperAngle = rightUpperArmAngle;
    float oldRightForeAngle = rightForeArmAngle;
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        rightUpperArmAngle += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        rightUpperArmAngle -= 1.0f;
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
        rightForeArmAngle += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
        rightForeArmAngle -= 1.0f;

    // 限制右臂角度
    rightUpperArmAngle = clampAngle(rightUpperArmAngle, -MAX_UPPER_ARM_ANGLE, -MIN_UPPER_ARM_ANGLE);
    rightForeArmAngle = clampAngle(rightForeArmAngle, -MAX_FORE_ARM_ANGLE, -MIN_FORE_ARM_ANGLE);

    // 攝像機控制
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        cameraRotation += 1.0f;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        cameraRotation -= 1.0f;
}

void drawRobot(const glm::mat4& projection, const glm::mat4& view) {
    glUseProgram(shaderProgram);
    
    // 設置光源和視點位置
    GLint lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos");
    GLint viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");
    glUniform3f(lightPosLoc, 2.0f, 3.0f, 2.0f);
    glUniform3f(viewPosLoc, 3.0f, 3.0f, 3.0f);

    // 設置變換矩陣
    GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");

    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

    // 身體基礎變換
    glm::mat4 bodyTransform = glm::mat4(1.0f);
    bodyTransform = glm::translate(bodyTransform, glm::vec3(0.0f, 0.0f, 0.0f));
    bodyTransform = glm::rotate(bodyTransform, glm::radians(bodyAngle), glm::vec3(0.0f, 1.0f, 0.0f));

    // 繪製身體
    glm::mat4 model = bodyTransform;
    model = glm::scale(model, glm::vec3(0.8f, 1.5f, 0.5f));  // 較大的身體
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 繪製頭部
    model = bodyTransform;
    model = glm::translate(model, glm::vec3(0.0f, 0.9f, 0.0f));
    model = glm::scale(model, glm::vec3(0.4f, 0.4f, 0.4f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 左臂
    glm::mat4 leftUpperArmTransform = bodyTransform;
    leftUpperArmTransform = glm::translate(leftUpperArmTransform, glm::vec3(-0.5f, 0.45f, 0.0f));
    leftUpperArmTransform = glm::rotate(leftUpperArmTransform, glm::radians(leftUpperArmAngle), glm::vec3(1.0f, 0.0f, 0.0f));
    
    model = leftUpperArmTransform;
    model = glm::scale(model, glm::vec3(0.2f, 0.6f, 0.2f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 左前臂 - 調整連接點
    glm::mat4 leftForeArmTransform = leftUpperArmTransform;
    leftForeArmTransform = glm::translate(leftForeArmTransform, glm::vec3(0.0f, -0.4f, 0.0f));  // 調整連接點
    leftForeArmTransform = glm::rotate(leftForeArmTransform, glm::radians(leftForeArmAngle), glm::vec3(1.0f, 0.0f, 0.0f));
    
    model = leftForeArmTransform;
    model = glm::scale(model, glm::vec3(0.15f, 0.3f, 0.15f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 右臂 - 調整初始位置和旋轉軸
    glm::mat4 rightUpperArmTransform = bodyTransform;
    rightUpperArmTransform = glm::translate(rightUpperArmTransform, glm::vec3(0.5f, 0.45f, 0.0f));
    rightUpperArmTransform = glm::rotate(rightUpperArmTransform, glm::radians(rightUpperArmAngle), glm::vec3(1.0f, 0.0f, 0.0f));
    
    model = rightUpperArmTransform;
    model = glm::scale(model, glm::vec3(0.2f, 0.6f, 0.2f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 右前臂 - 調整連接點
    glm::mat4 rightForeArmTransform = rightUpperArmTransform;
    rightForeArmTransform = glm::translate(rightForeArmTransform, glm::vec3(0.0f, -0.4f, 0.0f));  // 調整連接點
    rightForeArmTransform = glm::rotate(rightForeArmTransform, glm::radians(rightForeArmAngle), glm::vec3(1.0f, 0.0f, 0.0f));
    
    model = rightForeArmTransform;
    model = glm::scale(model, glm::vec3(0.15f, 0.3f, 0.15f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
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
        float radius = 5.0f;  // 增加攝像機距離
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
