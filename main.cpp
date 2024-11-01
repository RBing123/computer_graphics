#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <cmath>
#include <vector>
// 全局變量
float bodyAngle = 0.0f;       // 身體旋轉角度
float leftUpperArmAngle = 0.0f;   // 左上臂角度
float leftForeArmAngle = 0.0f;    // 左前臂角度
float rightUpperArmAngle = 0.0f;  // 右上臂角度
float rightForeArmAngle = 0.0f;   // 右前臂角度
float leftHipAngle = 0.0f;        // 左髖關節角度
float rightHipAngle = 0.0f;       // 右髖關節角度
float leftKneeAngle = 0.0f;   // 左膝關節角度
float rightKneeAngle = 0.0f;  // 右膝關節角度
float cameraRotation = 0.0f;      // 攝像機角度
float cameraHeight = 2.0f;
const float MAX_UPPER_ARM_ANGLE = 45.0f;   // 上臂最大角度
const float MIN_UPPER_ARM_ANGLE = -90.0f;  // 上臂最小角度
const float MAX_FORE_ARM_ANGLE = 45.0f;   // 前臂最大角度
const float MIN_FORE_ARM_ANGLE = 0.0f;    // 前臂最小角度
float leftUpperArmZAngle = 0.0f;   // 左上臂 Z 軸旋轉角度
float rightUpperArmZAngle = 0.0f;  // 右上臂 Z 軸旋轉角度
const float MAX_Z_ANGLE = 0.0f;   // Z 軸最大旋轉角度
const float MIN_Z_ANGLE = -90.0f;  // Z 軸最小旋轉角度
float leftFingerUpperAngle = -20.0f;   // 左手手指上段角度
float leftFingerLowerAngle = 0.0f;   // 左手手指下段角度
float rightFingerUpperAngle = -20.0f;  // 右手手指上段角度
float rightFingerLowerAngle = 0.0f;  // 右手手指下段角度
GLuint sphereVAO, sphereVBO, sphereEBO;
std::vector<float> sphereVertices;
std::vector<GLuint> sphereIndices;
GLuint cylinderVAO, cylinderVBO, cylinderEBO;
std::vector<float> cylinderVertices;
std::vector<GLuint> cylinderIndices;
GLuint coneVAO, coneVBO, coneEBO;
std::vector<float> coneVertices;
std::vector<GLuint> coneIndices;
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
    layout (location = 1) in vec3 aNormal;  // 添加法向量輸入

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform vec3 color;
    
    out vec3 FragPos;        // 傳遞片段位置
    out vec3 Normal;         // 傳遞法向量
    out vec3 fragColor;      // 傳遞顏色
    
    void main() {
        FragPos = vec3(model * vec4(aPos, 1.0));
        Normal = mat3(transpose(inverse(model))) * aNormal;  // 法向量變換
        fragColor = color;
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
)";

// 更新片段著色器
const char* fragmentShaderSource = R"(
    #version 330 core
    in vec3 FragPos;
    in vec3 Normal;
    in vec3 fragColor;
    
    uniform vec3 lightPos;        // 光源位置
    uniform vec3 viewPos;         // 攝像機位置
    
    out vec4 FragColor;
    
    void main() {
        // 環境光
        float ambientStrength = 0.2;
        vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
        
        // 漫反射
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
        
        // 鏡面反射
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
        
        vec3 result = (ambient + diffuse + specular) * fragColor;
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
void initSphere(float radius, int stacks, int sectors) {
    sphereVertices.clear();
    sphereIndices.clear();
    
    // 生成頂點
    float stackStep = M_PI / stacks;
    float sectorStep = 2 * M_PI / sectors;

    for (int i = 0; i <= stacks; ++i) {
        float phi = M_PI / 2 - i * stackStep;
        float xy = radius * cosf(phi);
        float z = radius * sinf(phi);

        for (int j = 0; j <= sectors; ++j) {
            float theta = j * sectorStep;
            float x = xy * cosf(theta);
            float y = xy * sinf(theta);
            
            // 添加頂點坐標
            sphereVertices.push_back(x);
            sphereVertices.push_back(y);
            sphereVertices.push_back(z);
        }
    }

    // 生成索引
    for (int i = 0; i < stacks; ++i) {
        int k1 = i * (sectors + 1);
        int k2 = k1 + sectors + 1;

        for (int j = 0; j < sectors; ++j, ++k1, ++k2) {
            if (i != 0) {
                sphereIndices.push_back(k1);
                sphereIndices.push_back(k2);
                sphereIndices.push_back(k1 + 1);
            }

            if (i != (stacks - 1)) {
                sphereIndices.push_back(k1 + 1);
                sphereIndices.push_back(k2);
                sphereIndices.push_back(k2 + 1);
            }
        }
    }

    // 創建並設置 VAO/VBO/EBO
    glGenVertexArrays(1, &sphereVAO);
    glGenBuffers(1, &sphereVBO);
    glGenBuffers(1, &sphereEBO);

    glBindVertexArray(sphereVAO);

    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(float), sphereVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndices.size() * sizeof(unsigned int), sphereIndices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}
void drawSphere() {
    glBindVertexArray(sphereVAO);
    glDrawElements(GL_TRIANGLES, sphereIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
void initCylinder(float radius, float height, int sectors) {
    cylinderVertices.clear();
    cylinderIndices.clear();

    // 生成頂部和底部圓的頂點
    for (int i = 0; i <= sectors; i++) {
        float angle = (float)i / sectors * 2.0f * M_PI;
        float x = radius * cos(angle);
        float z = radius * sin(angle);
        
        // 頂部圓頂點
        cylinderVertices.push_back(x);      // x
        cylinderVertices.push_back(height); // y
        cylinderVertices.push_back(z);      // z
        // 法線
        cylinderVertices.push_back(x/radius);
        cylinderVertices.push_back(0.0f);
        cylinderVertices.push_back(z/radius);
        
        // 底部圓頂點
        cylinderVertices.push_back(x);     // x
        cylinderVertices.push_back(0.0f);  // y
        cylinderVertices.push_back(z);     // z
        // 法線
        cylinderVertices.push_back(x/radius);
        cylinderVertices.push_back(0.0f);
        cylinderVertices.push_back(z/radius);
    }

    // 生成索引
    for (int i = 0; i < sectors; i++) {
        // 頂部圓的三角形
        cylinderIndices.push_back(i * 2);
        cylinderIndices.push_back((i + 1) * 2);
        cylinderIndices.push_back(sectors * 2 + 2); // 中心點

        // 底部圓的三角形
        cylinderIndices.push_back(i * 2 + 1);
        cylinderIndices.push_back((i + 1) * 2 + 1);
        cylinderIndices.push_back(sectors * 2 + 3); // 中心點

        // 側面的兩個三角形
        cylinderIndices.push_back(i * 2);
        cylinderIndices.push_back(i * 2 + 1);
        cylinderIndices.push_back((i + 1) * 2);

        cylinderIndices.push_back((i + 1) * 2);
        cylinderIndices.push_back(i * 2 + 1);
        cylinderIndices.push_back((i + 1) * 2 + 1);
    }

    // 添加頂部和底部中心點
    // 頂部中心點
    cylinderVertices.push_back(0.0f);    // x
    cylinderVertices.push_back(height);  // y
    cylinderVertices.push_back(0.0f);    // z
    cylinderVertices.push_back(0.0f);    // normal x
    cylinderVertices.push_back(1.0f);    // normal y
    cylinderVertices.push_back(0.0f);    // normal z

    // 底部中心點
    cylinderVertices.push_back(0.0f);   // x
    cylinderVertices.push_back(0.0f);   // y
    cylinderVertices.push_back(0.0f);   // z
    cylinderVertices.push_back(0.0f);   // normal x
    cylinderVertices.push_back(-1.0f);  // normal y
    cylinderVertices.push_back(0.0f);   // normal z

    // 創建並綁定 VAO、VBO、EBO
    glGenVertexArrays(1, &cylinderVAO);
    glGenBuffers(1, &cylinderVBO);
    glGenBuffers(1, &cylinderEBO);

    glBindVertexArray(cylinderVAO);

    glBindBuffer(GL_ARRAY_BUFFER, cylinderVBO);
    glBufferData(GL_ARRAY_BUFFER, cylinderVertices.size() * sizeof(float), 
                 cylinderVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cylinderEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cylinderIndices.size() * sizeof(GLuint), 
                 cylinderIndices.data(), GL_STATIC_DRAW);

    // 設置頂點屬性指針
    // 位置屬性
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // 法線屬性
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), 
                         (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}
void drawCylinder() {
    glBindVertexArray(cylinderVAO);
    glDrawElements(GL_TRIANGLES, cylinderIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
void initCone(float bottomRadius, float topRadius, float height, int segments) {
    coneVertices.clear();
    coneIndices.clear();

    // 生成頂點
    // 底部圓的頂點
    for(int i = 0; i <= segments; i++) {
        float theta = 2.0f * M_PI * i / segments;
        float x = bottomRadius * cos(theta);
        float z = bottomRadius * sin(theta);
        
        // 位置
        coneVertices.push_back(x);      // x
        coneVertices.push_back(0.0f);   // y
        coneVertices.push_back(z);      // z
        // 法向量
        float nx = x / bottomRadius;
        float nz = z / bottomRadius;
        float ny = (bottomRadius - topRadius) / height;
        float len = sqrt(nx*nx + ny*ny + nz*nz);
        coneVertices.push_back(nx/len);
        coneVertices.push_back(ny/len);
        coneVertices.push_back(nz/len);
    }

    // 頂部圓的頂點
    for(int i = 0; i <= segments; i++) {
        float theta = 2.0f * M_PI * i / segments;
        float x = topRadius * cos(theta);
        float z = topRadius * sin(theta);
        
        // 位置
        coneVertices.push_back(x);      // x
        coneVertices.push_back(height); // y
        coneVertices.push_back(z);      // z
        // 法向量
        float nx = x / topRadius;
        float nz = z / topRadius;
        float ny = (bottomRadius - topRadius) / height;
        float len = sqrt(nx*nx + ny*ny + nz*nz);
        coneVertices.push_back(nx/len);
        coneVertices.push_back(ny/len);
        coneVertices.push_back(nz/len);
    }

    // 生成索引
    for(int i = 0; i < segments; i++) {
        coneIndices.push_back(i);
        coneIndices.push_back(i + 1);
        coneIndices.push_back(i + segments + 1);

        coneIndices.push_back(i + segments + 1);
        coneIndices.push_back(i + 1);
        coneIndices.push_back(i + segments + 2);
    }

    // 設置 VAO、VBO 和 EBO
    glGenVertexArrays(1, &coneVAO);
    glGenBuffers(1, &coneVBO);
    glGenBuffers(1, &coneEBO);

    glBindVertexArray(coneVAO);

    glBindBuffer(GL_ARRAY_BUFFER, coneVBO);
    glBufferData(GL_ARRAY_BUFFER, coneVertices.size() * sizeof(float), coneVertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, coneEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, coneIndices.size() * sizeof(GLuint), coneIndices.data(), GL_STATIC_DRAW);

    // 位置屬性
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // 法向量屬性
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}
void drawCone() {
    glBindVertexArray(coneVAO);
    glDrawElements(GL_TRIANGLES, coneIndices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}
void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    
    float rotationSpeed = 2.0f;  // 可以調整旋轉速度
    float heightSpeed = 0.1f;    // 上下移動速度
    // 身體旋轉
    // if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    //     bodyAngle += rotationSpeed;
    // if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
    //     bodyAngle -= rotationSpeed;
    
    // 左手臂控制
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        leftUpperArmAngle += rotationSpeed;  // 向後抬
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
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
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
        rightForeArmAngle += rotationSpeed;   // 手肘彎曲
    if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
        rightForeArmAngle -= rotationSpeed;   // 手肘伸直
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
        leftUpperArmZAngle += rotationSpeed;
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
        leftUpperArmZAngle -= rotationSpeed;

    // 添加右臂 Z 軸旋轉控制
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
        rightUpperArmZAngle += rotationSpeed;
    if (glfwGetKey(window, GLFW_KEY_Y) == GLFW_PRESS)
        rightUpperArmZAngle -= rotationSpeed;

    // 限制 Z 軸旋轉角度
    leftUpperArmZAngle = clampAngle(leftUpperArmZAngle, MIN_Z_ANGLE, MAX_Z_ANGLE);
    rightUpperArmZAngle = clampAngle(rightUpperArmZAngle, MIN_Z_ANGLE, MAX_Z_ANGLE);
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
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        cameraHeight += heightSpeed;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        cameraHeight -= heightSpeed;
    float fingerSpeed = 2.0f;  // 手指旋轉速度

    // 左手手指控制
    // 上段控制
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
        leftFingerUpperAngle += fingerSpeed;  // 張開
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
        leftFingerUpperAngle -= fingerSpeed;  // 夾緊

    // 下段控制
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS)
        leftFingerLowerAngle += fingerSpeed;  // 張開
    if (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS)
        leftFingerLowerAngle -= fingerSpeed;  // 夾緊

    // 右手手指控制
    // 上段控制
    if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS)
        rightFingerUpperAngle += fingerSpeed;  // 張開
    if (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS)
        rightFingerUpperAngle -= fingerSpeed;  // 夾緊

    // 下段控制
    if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS)
        rightFingerLowerAngle += fingerSpeed;  // 張開
    if (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS)
        rightFingerLowerAngle -= fingerSpeed;  // 夾緊

    // 限制角度範圍
    leftFingerUpperAngle = glm::clamp(leftFingerUpperAngle, -45.0f, -10.0f);
    leftFingerLowerAngle = glm::clamp(leftFingerLowerAngle, 5.0f, 45.0f);
    rightFingerUpperAngle = glm::clamp(rightFingerUpperAngle, -45.0f, -10.0f);
    rightFingerLowerAngle = glm::clamp(rightFingerLowerAngle, 5.0f, 45.0f);
    // 添加髖關節控制
    if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
        leftHipAngle += rotationSpeed;
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS)
        leftHipAngle -= rotationSpeed;
        
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
        rightHipAngle += rotationSpeed;
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
        rightHipAngle -= rotationSpeed;

    // 限制髖關節角度範圍
    leftHipAngle = clampAngle(leftHipAngle, -45.0f, 45.0f);
    rightHipAngle = clampAngle(rightHipAngle, -45.0f, 45.0f);
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS)
        leftKneeAngle -= rotationSpeed;  // 往前彎曲是負角度
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
        leftKneeAngle += rotationSpeed;  // 往後伸直是正角度
        
    if (glfwGetKey(window, GLFW_KEY_COMMA) == GLFW_PRESS)
        rightKneeAngle -= rotationSpeed;
    if (glfwGetKey(window, GLFW_KEY_PERIOD) == GLFW_PRESS)
        rightKneeAngle += rotationSpeed;

    // 限制膝蓋角度
    leftKneeAngle = clampAngle(leftKneeAngle, 0.0f, 90.0f);   // 膝蓋只能向後彎曲
    rightKneeAngle = clampAngle(rightKneeAngle, 0.0f, 90.0f);
    // 限制攝影機高度範圍
    cameraHeight = glm::clamp(cameraHeight, -0.5f, 5.0f);
}

void drawRobot(const glm::mat4& projection, const glm::mat4& view) {
    glUseProgram(shaderProgram);
    
    // 獲取uniform位置
    GLuint modelLoc = glGetUniformLocation(shaderProgram, "model");
    GLuint viewLoc = glGetUniformLocation(shaderProgram, "view");
    GLuint projLoc = glGetUniformLocation(shaderProgram, "projection");
    GLuint colorLoc = glGetUniformLocation(shaderProgram, "color");
    GLuint lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos");
    GLuint viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");

    // 設置光源位置和攝像機位置
    glm::vec3 lightPos(1.0f, 1.0f, 1.0f);  // 光源位置
    glm::vec3 viewPos(0.0f, 0.0f, 3.0f);   // 攝像機位置
    
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniform3fv(lightPosLoc, 1, glm::value_ptr(lightPos));
    glUniform3fv(viewPosLoc, 1, glm::value_ptr(viewPos));

    // 基礎變換
    glm::mat4 baseTransform = glm::mat4(1.0f);
    baseTransform = glm::translate(baseTransform, glm::vec3(0.0f, 0.0f, 0.0f));
    baseTransform = glm::rotate(baseTransform, glm::radians(bodyAngle), glm::vec3(0.0f, 1.0f, 0.0f));

    glBindVertexArray(VAO);  // 綁定立方體的VAO
    
    // 1. 身體核心部分
    // 上身軀幹（胸甲）
    glm::mat4 model = baseTransform;
    model = glm::translate(model, glm::vec3(0.0f, 0.6f, 0.0f));
    model = glm::scale(model, glm::vec3(0.6f, 0.6f, 0.3f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.45f, 0.329f, 0.26f);  // 深藍色胸甲
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 胸部中央裝甲（白色部分）
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.0f, 0.6f, 0.16f));
    model = glm::scale(model, glm::vec3(0.4f, 0.4f, 0.01f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.9f, 0.9f, 0.9f);  // 白色裝甲
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 頸部（圓柱形）
    glBindVertexArray(cylinderVAO);
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.0f, 0.9f, 0.0f));  // 位置在胸甲上方
    model = glm::scale(model, glm::vec3(0.05f, 0.08f, 0.05f));    // 調整圓柱體的大小
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.6f, 0.6f, 0.6f);  // 灰色頸部
    drawCylinder();
    // 頭部（橢圓形）
    glBindVertexArray(sphereVAO);
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.0f, 1.05f, 0.0f));    // 位置在頸部上方
    model = glm::scale(model, glm::vec3(1.0f, 1.1f, 1.0f));      // 橢圓形狀的縮放
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.1f, 0.1f, 0.1f);
    drawSphere();
    // 左眼
    glBindVertexArray(sphereVAO);  // 使用球體來創造圓形眼睛
    model = baseTransform;
    model = glm::translate(model, glm::vec3(-0.05f, 1.05f, 0.05f));
    // 使用不同的縮放比例創造橢圓形
    model = glm::scale(model, glm::vec3(0.3f, 0.2f, 0.5f));  // x寬度大，y高度小，z適中
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 1.0f, 0.8f, 0.0f);  // 橙黃色
    drawSphere();
    // 右眼
    glBindVertexArray(sphereVAO);  // 使用球體來創造圓形眼睛
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.05f, 1.05f, 0.05f));
    // 使用不同的縮放比例創造橢圓形
    model = glm::scale(model, glm::vec3(0.3f, 0.2f, 0.5f));  // x寬度大，y高度小，z適中
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 1.0f, 0.8f, 0.0f);  // 橙黃色
    drawSphere();
    // 主天線（圓柱形）
    glBindVertexArray(cylinderVAO);
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.0f, 1.1f, 0.0f));    // 位置在頭頂
    model = glm::scale(model, glm::vec3(0.01f, 0.15f, 0.01f));     // 細長的圓柱體
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.7f, 0.7f, 0.7f);  // 銀灰色天線
    drawCylinder();

    // 天線頂部小球
    glBindVertexArray(sphereVAO);
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.0f, 1.25f, 0.0f));   // 位置在天線頂端
    model = glm::scale(model, glm::vec3(0.2f, 0.2f, 0.2f));     // 小球體
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 1.0f, 0.5f, 0.0f);
    drawSphere();

    // mouth
    glBindVertexArray(VAO);
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.0f, 1.0f, 0.04f));
    model = glm::scale(model, glm::vec3(0.1f, 0.007f, 0.12f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 1.0f, 0.6f, 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    // 2. 肩部裝甲
    glBindVertexArray(VAO);
    // 左肩甲
    model = baseTransform;
    model = glm::translate(model, glm::vec3(-0.35f, 0.8f, 0.0f));
    model = glm::scale(model, glm::vec3(0.3f, 0.25f, 0.25f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.568f, 0.568f, 0.317f);  // 橙色肩甲
    glDrawArrays(GL_TRIANGLES, 0, 36);
    // 左肩裝甲板
    model = baseTransform;
    model = glm::translate(model, glm::vec3(-0.55f, 0.8f, 0.0f));
    model = glm::scale(model, glm::vec3(0.1f, 0.4f, 0.3f));  // 扁平的長方形
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.4f, 0.4f, 0.4f);  // 深灰色裝甲
    glDrawArrays(GL_TRIANGLES, 0, 36);
    // 左肩關節（球形）
    glBindVertexArray(sphereVAO);
    glm::mat4 leftShoulderTransform = baseTransform;
    leftShoulderTransform = glm::translate(leftShoulderTransform, glm::vec3(-0.4f, 0.65f, 0.0f));
    model = leftShoulderTransform;
    model = glm::scale(model, glm::vec3(0.45f, 0.45f, 0.45f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.8f, 0.2f, 0.2f);
    drawSphere();

    // 左上臂 - 使用與前臂相同的邏輯
    glBindVertexArray(VAO);
    glm::mat4 leftUpperArmTransform = leftShoulderTransform;
    leftUpperArmTransform = glm::rotate(leftUpperArmTransform, 
        glm::radians(leftUpperArmZAngle), glm::vec3(0.0f, 0.0f, 1.0f));
    // 先進行旋轉
    leftUpperArmTransform = glm::rotate(leftUpperArmTransform, 
        glm::radians(leftUpperArmAngle), glm::vec3(1.0f, 0.0f, 0.0f));

    // 修改上臂繪製方式
    model = leftUpperArmTransform;
    // 不再需要向下位移，直接從關節中心開始畫
    model = glm::scale(model, glm::vec3(0.1f, 0.3f, 0.1f));  // 先縮放
    model = glm::translate(model, glm::vec3(0.0f, -0.65f, 0.0f));  // 再向下位移使上臂從中心向下延伸
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.54f, 0.55f, 0.545f);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    glBindVertexArray(sphereVAO);
    // 左肘關節（球形）
    model = leftUpperArmTransform;
    model = glm::translate(model, glm::vec3(0.0f, -0.4f, 0.0f));  // 調整關節位置
    model = glm::scale(model, glm::vec3(0.55f, 0.55f, 0.55f));  // 稍微縮小關節
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.8f, 0.0f, 0.0f);
    drawSphere();

    glBindVertexArray(VAO);

    // 左前臂
    glm::mat4 leftForeArmTransform = leftUpperArmTransform;
    leftForeArmTransform = glm::translate(leftForeArmTransform, glm::vec3(0.0f, -0.4f, 0.0f));
    leftForeArmTransform = glm::rotate(leftForeArmTransform, glm::radians(leftForeArmAngle), glm::vec3(-1.0f, 0.0f, 0.0f));

    float sphereRadius = 0.55f * 0.1f;
    float forearmOffsetY = -sphereRadius * cos(glm::radians(leftForeArmAngle));
    float forearmOffsetZ = sphereRadius * sin(glm::radians(leftForeArmAngle));
    leftForeArmTransform = glm::translate(leftForeArmTransform, 
    glm::vec3(0.0f, forearmOffsetY, forearmOffsetZ));
    leftForeArmTransform = glm::rotate(leftForeArmTransform, glm::radians(leftForeArmAngle), glm::vec3(-1.0f, 0.0f, 0.0f));

    
    // 左手手指
    // 第一根手指（左側）
    // 上段
    glBindVertexArray(VAO);
    model = leftForeArmTransform;
    model = glm::translate(model, glm::vec3(-0.05f, -0.35f, 0.0f)); 
    model = glm::translate(model, glm::vec3(0.0f, 0.05f, 0.0f));    
    model = glm::rotate(model, glm::radians(leftFingerUpperAngle), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::translate(model, glm::vec3(0.0f, -0.05f, 0.0f));   
    model = glm::scale(model, glm::vec3(0.04f, 0.1f, 0.04f));      
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.45f, 0.26f, 0.07f);  
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 下段
    model = leftForeArmTransform;
    model = glm::translate(model, glm::vec3(-0.05f, -0.35f, 0.0f));
    model = glm::translate(model, glm::vec3(0.0f, 0.05f, 0.0f));
    model = glm::rotate(model, glm::radians(leftFingerUpperAngle), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::translate(model, glm::vec3(0.0f, -0.15f, 0.0f));
    model = glm::translate(model, glm::vec3(0.0f, 0.04f, 0.0f));
    model = glm::rotate(model, glm::radians(leftFingerLowerAngle), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::translate(model, glm::vec3(0.0f, -0.04f, 0.0f));
    model = glm::scale(model, glm::vec3(0.04f, 0.08f, 0.04f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.55f, 0.74f, 0.93f);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 第二根手指（左側）
    // 上段
    glBindVertexArray(VAO);
    model = leftForeArmTransform;
    model = glm::translate(model, glm::vec3(0.05f, -0.35f, 0.0f)); 
    model = glm::translate(model, glm::vec3(0.0f, 0.05f, 0.0f));    
    model = glm::rotate(model, glm::radians(5.0f - leftFingerUpperAngle), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::translate(model, glm::vec3(0.0f, -0.05f, 0.0f));   
    model = glm::scale(model, glm::vec3(0.04f, 0.1f, 0.04f));      
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.45f, 0.26f, 0.07f);  
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 下段
    model = leftForeArmTransform;
    model = glm::translate(model, glm::vec3(0.05f, -0.35f, 0.0f));
    model = glm::translate(model, glm::vec3(0.0f, 0.05f, 0.0f));
    model = glm::rotate(model, glm::radians(5.0f - leftFingerUpperAngle), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::translate(model, glm::vec3(0.0f, -0.15f, 0.0f));
    model = glm::translate(model, glm::vec3(0.0f, 0.04f, 0.0f));
    model = glm::rotate(model, glm::radians(-leftFingerLowerAngle), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::translate(model, glm::vec3(0.0f, -0.04f, 0.0f));
    model = glm::scale(model, glm::vec3(0.04f, 0.08f, 0.04f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.55f, 0.74f, 0.93f);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 右肩
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.35f, 0.8f, 0.0f));
    model = glm::scale(model, glm::vec3(0.3f, 0.25f, 0.25f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.568f, 0.568f, 0.317f);  // 橙色肩甲
    glDrawArrays(GL_TRIANGLES, 0, 36);
    // 左肩裝甲板
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.55f, 0.8f, 0.0f));
    model = glm::scale(model, glm::vec3(0.1f, 0.4f, 0.3f));  // 扁平的長方形
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.4f, 0.4f, 0.4f);  // 深灰色裝甲
    glDrawArrays(GL_TRIANGLES, 0, 36);
    // joint
    glBindVertexArray(sphereVAO);
    glm::mat4 rightShoulderTransform = baseTransform;
    rightShoulderTransform = glm::translate(rightShoulderTransform, glm::vec3(0.4f, 0.65f, 0.0f));  // 改為正的x值
    model = rightShoulderTransform;
    model = glm::scale(model, glm::vec3(0.45f, 0.45f, 0.45f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.8f, 0.0f, 0.0f);
    drawSphere();

    glBindVertexArray(VAO);
    // 繪製前臂
    model = leftForeArmTransform;
    model = glm::translate(model, glm::vec3(0.0f, -0.15f, 0.0f));
    model = glm::scale(model, glm::vec3(0.15f, 0.3f, 0.15f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.0f, 0.8f, 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    // 右上臂
    glm::mat4 rightUpperArmTransform = rightShoulderTransform;
    rightUpperArmTransform = glm::rotate(rightUpperArmTransform, 
        glm::radians(rightUpperArmZAngle), glm::vec3(0.0f, 0.0f, -1.0f));
    // 先進行旋轉
    rightUpperArmTransform = glm::rotate(rightUpperArmTransform, 
        glm::radians(rightUpperArmAngle), glm::vec3(1.0f, 0.0f, 0.0f));

    // 修改上臂繪製方式
    model = rightUpperArmTransform;
    // 不再需要向下位移，直接從關節中心開始畫
    model = glm::scale(model, glm::vec3(0.1f, 0.3f, 0.1f));  // 先縮放
    model = glm::translate(model, glm::vec3(0.0f, -0.65f, 0.0f));  // 再向下位移使上臂從中心向下延伸
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.54f, 0.55f, 0.545f);  // 銀色上臂
    glDrawArrays(GL_TRIANGLES, 0, 36);

    glBindVertexArray(sphereVAO);
    // 右肘關節（球形）
    model = rightUpperArmTransform;
    model = glm::translate(model, glm::vec3(0.0f, -0.4f, 0.0f));  // 調整關節位置
    model = glm::scale(model, glm::vec3(0.55f, 0.55f, 0.55f));  // 稍微縮小關節
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.8f, 0.0f, 0.0f);
    drawSphere();

    glBindVertexArray(VAO);
    // 右前臂
    glm::mat4 rightForeArmTransform = rightUpperArmTransform;
    rightForeArmTransform = glm::translate(rightForeArmTransform, glm::vec3(0.0f, -0.4f, 0.0f));
    rightForeArmTransform = glm::rotate(rightForeArmTransform, glm::radians(rightForeArmAngle), glm::vec3(-1.0f, 0.0f, 0.0f));
    
    float rforearmOffsetY = -sphereRadius * cos(glm::radians(rightForeArmAngle));
    float rforearmOffsetZ = sphereRadius * sin(glm::radians(rightForeArmAngle));
    rightForeArmTransform = glm::translate(rightForeArmTransform, 
    glm::vec3(0.0f, rforearmOffsetY, rforearmOffsetZ));
    rightForeArmTransform = glm::rotate(rightForeArmTransform, glm::radians(rightForeArmAngle), glm::vec3(-1.0f, 0.0f, 0.0f));
    
    // 繪製前臂
    model = rightForeArmTransform;
    model = glm::translate(model, glm::vec3(0.0f, -0.15f, 0.0f));
    model = glm::scale(model, glm::vec3(0.15f, 0.3f, 0.15f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.0f, 0.8f, 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    // 右手手指
    // 第一根手指（左側）
    // 上段
    glBindVertexArray(VAO);
    model = rightForeArmTransform;
    model = glm::translate(model, glm::vec3(-0.05f, -0.35f, 0.0f));      
    model = glm::translate(model, glm::vec3(0.0f, 0.05f, 0.0f));    
    model = glm::rotate(model, glm::radians(rightFingerUpperAngle), glm::vec3(0.0f, 0.0f, 1.0f));  // 初始向外偏轉
    model = glm::translate(model, glm::vec3(0.0f, -0.05f, 0.0f));   
    model = glm::scale(model, glm::vec3(0.04f, 0.1f, 0.04f));      
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.45f, 0.26f, 0.07f);  
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 下段
    model = rightForeArmTransform;
    model = glm::translate(model, glm::vec3(-0.05f, -0.35f, 0.0f));      
    model = glm::translate(model, glm::vec3(0.0f, 0.05f, 0.0f));
    model = glm::rotate(model, glm::radians(rightFingerUpperAngle), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::translate(model, glm::vec3(0.0f, -0.15f, 0.0f));
    model = glm::translate(model, glm::vec3(0.0f, 0.04f, 0.0f));
    model = glm::rotate(model, glm::radians(rightFingerLowerAngle), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::translate(model, glm::vec3(0.0f, -0.04f, 0.0f));
    model = glm::scale(model, glm::vec3(0.04f, 0.08f, 0.04f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.55f, 0.74f, 0.93f);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 第二根手指（右側）
    // 上段
    model = rightForeArmTransform;
    model = glm::translate(model, glm::vec3(0.05f, -0.35f, 0.0f));      
    model = glm::translate(model, glm::vec3(0.0f, 0.05f, 0.0f));    
    model = glm::rotate(model, glm::radians(5.0f - rightFingerUpperAngle), glm::vec3(0.0f, 0.0f, 1.0f));  
    model = glm::translate(model, glm::vec3(0.0f, -0.05f, 0.0f));   
    model = glm::scale(model, glm::vec3(0.04f, 0.1f, 0.04f));      
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.45f, 0.26f, 0.07f);  
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 下段
    model = rightForeArmTransform;
    model = glm::translate(model, glm::vec3(0.05f, -0.35f, 0.0f));      
    model = glm::translate(model, glm::vec3(0.0f, 0.05f, 0.0f));
    model = glm::rotate(model, glm::radians(5.0f - rightFingerUpperAngle - 5.0f), glm::vec3(0.0f, 0.0f, 1.0f));  
    model = glm::translate(model, glm::vec3(0.0f, -0.15f, 0.0f));
    model = glm::translate(model, glm::vec3(0.0f, 0.04f, 0.0f));
    model = glm::rotate(model, glm::radians(-rightFingerLowerAngle), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::translate(model, glm::vec3(0.0f, -0.04f, 0.0f));
    model = glm::scale(model, glm::vec3(0.04f, 0.08f, 0.04f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.55f, 0.74f, 0.93f);
    glDrawArrays(GL_TRIANGLES, 0, 36);

    // 左髖關節
    glBindVertexArray(sphereVAO);
    model = baseTransform;
    model = glm::translate(model, glm::vec3(-0.15f, 0.25f, 0.0f));    // 位於身體下方左側
    model = glm::scale(model, glm::vec3(0.45f, 0.45f, 0.45f));      // 與肩關節相似的大小
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.8f, 0.0f, 0.0f);  // 紅色關節
    drawSphere();

    // 右髖關節
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.15f, 0.25f, 0.0f));     // 位於身體下方右側
    model = glm::scale(model, glm::vec3(0.45f, 0.45f, 0.45f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.8f, 0.0f, 0.0f);
    drawSphere();
    // 創建左右髖關節的變換矩陣（用於連接大腿）
    glm::mat4 leftHipTransform = baseTransform;
    leftHipTransform = glm::translate(leftHipTransform, glm::vec3(-0.15f, 0.25f, 0.0f));
    leftHipTransform = glm::rotate(leftHipTransform, glm::radians(leftHipAngle), glm::vec3(1.0f, 0.0f, 0.0f));

    glm::mat4 rightHipTransform = baseTransform;
    rightHipTransform = glm::translate(rightHipTransform, glm::vec3(0.15f, 0.25f, 0.0f));
    rightHipTransform = glm::rotate(rightHipTransform, glm::radians(rightHipAngle), glm::vec3(1.0f, 0.0f, 0.0f));

    // 左大腿（圓柱體）
    glBindVertexArray(cylinderVAO);
    model = leftHipTransform;
    glm::mat4 leftLegTransform = model;
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));  // X軸旋轉90度
    model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.05f));    // 向下移動到關節位置
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)); // 旋轉圓柱體使其垂直
    model = glm::scale(model, glm::vec3(0.1f, 0.5f, 0.1f));       // 調整大腿的粗細和長度
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.54f, 0.55f, 0.545f);  // 綠色大腿
    drawCylinder();

    
    // 左腿變換矩陣
    glm::mat4 leftKneeTransform = leftHipTransform;
    leftKneeTransform = glm::translate(leftKneeTransform, glm::vec3(0.0f, -0.6f, 0.0f));  // 移動到大腿末端
    leftKneeTransform = glm::rotate(leftKneeTransform, glm::radians(leftKneeAngle), glm::vec3(1.0f, 0.0f, 0.0f));  // 膝蓋旋轉
    // 左膝蓋關節
    glBindVertexArray(sphereVAO);
    model = leftLegTransform;  // 使用儲存的變換
    model = glm::translate(model, glm::vec3(0.0f, -0.6f, 0.0f));     // 與其他關節相同大小
    model = glm::scale(model, glm::vec3(0.5f, 0.5f, 0.5f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.8f, 0.0f, 0.0f);  // 紅色關節
    drawSphere();
    
    
    // 右大腿（圓柱體）
    model = rightHipTransform;
    glm::mat4 rightLegTransform = model;
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));  // X軸旋轉90度
    model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.05f));    
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::scale(model, glm::vec3(0.1f, 0.5f, 0.1f));       
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.54f, 0.55f, 0.545f);
    drawCylinder();
    
    // 右腿變換矩陣
    glm::mat4 rightKneeTransform = rightHipTransform;
    rightKneeTransform = glm::translate(rightKneeTransform, glm::vec3(0.0f, -0.6f, 0.0f));  // 移動到大腿末端
    rightKneeTransform = glm::rotate(rightKneeTransform, glm::radians(rightKneeAngle), glm::vec3(1.0f, 0.0f, 0.0f));  // 膝蓋旋轉
    // 右膝蓋關節
    glBindVertexArray(sphereVAO);
    model = rightLegTransform;  // 使用儲存的變換
    model = glm::translate(model, glm::vec3(0.0f, -0.6f, 0.0f));     // 與其他關節相同大小
    model = glm::scale(model, glm::vec3(0.5f, 0.5f, 0.5f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.8f, 0.0f, 0.0f);  // 紅色關節
    drawSphere();

    // 左小腿
    glBindVertexArray(coneVAO);
    model = leftKneeTransform;
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.03f));    // 從膝蓋關節向下
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)); // 使圓錐台垂直
    model = glm::scale(model, glm::vec3(0.08f, 0.4f, 0.08f));       // 調整大小
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.407f, 0.509f, 0.549f);  // 與大腿相同的綠色
    drawCone();
    // left feet
    model = leftKneeTransform;
    model = glm::translate(model, glm::vec3(0.0f, -0.42f, 0.0f));    // 移動到小腿底部並稍微向前
    model = glm::rotate(model, glm::radians(-180.0f), glm::vec3(1.0f, 0.0f, 0.0f));  // 水平放置
    model = glm::scale(model, glm::vec3(0.09f, 0.09f, 0.09f));      // 較小的尺寸
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.7f, 0.7f, 0.7f);  // 銀色腳掌
    drawCone();
    // 右小腿
    model = rightKneeTransform;
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.03f));
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    model = glm::scale(model, glm::vec3(0.08f, 0.4f, 0.08f));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.407f, 0.509f, 0.549f);
    drawCone();
    // right feet
    model = rightKneeTransform;
    model = glm::translate(model, glm::vec3(0.0f, -0.42f, 0.0f));    // 移動到小腿底部並稍微向前
    model = glm::rotate(model, glm::radians(-180.0f), glm::vec3(1.0f, 0.0f, 0.0f));  // 水平放置
    model = glm::scale(model, glm::vec3(0.09f, 0.09f, 0.09f));      // 較小的尺寸
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.7f, 0.7f, 0.7f);  // 銀色腳掌
    drawCone();

    // decloration
    // 炮塔基座（扁平的圓柱體）
    glBindVertexArray(cylinderVAO);
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.0f, 0.65f, -0.4f));  // 位於背部中央偏上
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f)); // 使圓柱體平躺
    model = glm::scale(model, glm::vec3(0.25f, 0.3f, 0.25f));      // 扁平的圓盤形狀
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.85f, 0.76f, 0.49f);  // 深灰色
    drawCylinder();

    // 主炮（圓柱體）
    model = baseTransform;
    model = glm::translate(model, glm::vec3(0.0f, 1.2f, -0.35f));  // 起始位置
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));  // 轉向
    model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.2f));    // 向前延伸
    // 確保 X 和 Z 軸縮放相同，保持圓形橫截面
    model = glm::scale(model, glm::vec3(0.08f, 0.08f, 0.4f));    // 調整長度，保持圓形截面
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.7f, 0.7f, 0.7f);
    drawCylinder();
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

    initSphere(0.1f, 20, 20);
    createCube();
    initCylinder(1.0f, 1.0f, 32);
    initCone(0.7f, 1.5f, 1.0f, 32);

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
            glm::vec3(camX, cameraHeight, camZ),  // 使用cameraHeight作為Y值
            glm::vec3(0.0f, 0.0f, 0.0f),         // 始終看向原點
            glm::vec3(0.0f, 1.0f, 0.0f)          // 上向量保持不變
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
    glDeleteVertexArrays(1, &sphereVAO);
    glDeleteBuffers(1, &sphereVBO);
    glDeleteBuffers(1, &sphereEBO);
    glDeleteProgram(shaderProgram);

    glfwTerminate();
    return 0;
}
